import os.path as osp
import importlib
import json
import copy
from pathlib import Path
from typing import List
import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder
from mani_skill.utils.structs.pose import Pose

from ..scene_builder import SceneBuilder, RFSceneBuilder

class TableSceneBuilder(RFSceneBuilder):
    def build(self):
        # scene
        scene_cfg = self.cfg['scene']
        altitude = 0
        self.scene_objects = {}
        if 'primitives' in scene_cfg:
            for primitive_cfg in scene_cfg['primitives']:
                primitive_name = primitive_cfg['name']
                builder_module_name, builder_class_name = primitive_cfg['builder'].rsplit('.', maxsplit=1)
                builder_module = importlib.import_module(builder_module_name)
                builder = getattr(builder_module, builder_class_name)
                params = primitive_cfg['params']
                if 'initial_pose' in params:
                    params['initial_pose'] = sapien.Pose(p=params['initial_pose']['p'])
                primitive = builder(self.env.scene, **params)
                setattr(self.env, primitive_name, primitive)
                self.scene_objects[primitive_name] = getattr(self.env, primitive_name)
        if 'assets' in scene_cfg:        
            for asset_cfg in scene_cfg['assets']:
                asset_file = asset_cfg['file_path']
                asset_type = osp.splitext(asset_file)[-1]
                if asset_type in ['.obj', '.glb']:
                    builder = self.scene.create_actor_builder()
                    asset_name = asset_cfg['name']
                    # TODO: update different collision types
                    if asset_cfg['collision']['type'] == 'box':
                        builder.add_box_collision(
                            pose=sapien.Pose(p=asset_cfg['collision']['pos']['p']),
                            half_size=asset_cfg['collision']['pos']['half_size'],
                    )
                    temp_pose = sapien.Pose(q=euler2quat(*asset_cfg['pos']['ppos']['q']))
                    builder.add_visual_from_file(
                        filename=asset_file, scale=asset_cfg['scale'], pose=temp_pose
                    )
                    initial_ppos = asset_cfg['pos']['ppos']['p']
                    if 'randp_scale' in asset_cfg['pos']:
                        initial_ppos = np.array(initial_ppos) + np.array(asset_cfg['pos']['randp_scale']) * np.random.rand((len(initial_ppos)))
                        initial_ppos = initial_ppos.tolist()
                    builder.initial_pose = sapien.Pose(
                        p=asset_cfg['pos']['ppos']['p'], q=euler2quat(*asset_cfg['pos']['ppos']['q'])
                    )
                    setattr(self.env, asset_name, builder.build_kinematic(name=f"{scene_cfg['name']}-Workspace"))
                    aabb = (
                        getattr(self.env, asset_name)._objs[0]
                        .find_component_by_type(sapien.render.RenderBodyComponent)
                        .compute_global_aabb_tight()
                    )
                    height = aabb[1, 2] - aabb[0, 2]
                    altitude = min(altitude, -height)    # make the plane at 0 in z
                    self.scene_objects[asset_name] = getattr(self.env, asset_name)
                elif asset_type in ['.urdf']:
                    pass
        self.ground = build_ground(
            self.scene, floor_width=scene_cfg['env']['floor_width'], altitude=altitude
        )
        self.scene_objects['ground'] = self.ground

        # objects
        if 'objects' in self.cfg:
            objects_cfg = self.cfg['objects']
            self.movable_objects = {}
            for object_cfg in objects_cfg:
                object_file_path = object_cfg['file_path']
                object_type = osp.splitext(object_file_path)[-1]
                object_name = object_cfg['name']
                object_annotation_path = object_cfg['annotation_path']
                with open(object_annotation_path, 'r') as f:
                    object_annotation_data = json.load(f)
                self.env.annotation_data[object_name] = object_annotation_data
                if object_type in ['.obj', '.glb']:
                    builder = self.scene.create_actor_builder()
                    builder.set_physx_body_type("dynamic")
                    visual_params = {}
                    collision_params = {}
                    if 'material' in object_cfg:
                        physx, material = object_cfg['material']['type'].rsplit('.', maxsplit=1)
                        physx_module = importlib.import_module(physx)
                        material_builder = getattr(physx_module, material)
                        object_material = material_builder(**object_cfg['material']['params'])
                        collision_params['material'] = object_material
                    visual_cfg = object_cfg['visual']
                    visual_params.update(visual_cfg)
                    if object_cfg.get('collision', None):
                        collision_params.update(object_cfg['collision'])
                    else:
                        collision_params.update(visual_params)    # use visual cfg as default
                    if collision_params['type'] == 'nonconvex':
                        del(collision_params['type'])
                        builder.add_nonconvex_collision_from_file(**collision_params)
                    else:
                        del(collision_params['type'])
                        builder.add_convex_collision_from_file(**collision_params)
                    builder.add_visual_from_file(**visual_params)
                    if object_cfg.get('mass_params', None):
                        mass_params = object_cfg['mass_params']
                        if 'cmass_local_pose' in mass_params:
                            mass_params['cmass_local_pose'] = sapien.Pose(mass_params['cmass_local_pose'])
                        builder.set_mass_and_inertia(**mass_params)
                    setattr(self.env, object_name, builder.build(name=object_name))
                elif object_type in ['.urdf']:
                    # use nonconvex collision
                    def create_nonconvex_urdf_loader(scene):
                        from utils.building.nonconvex_urdf_loader import NonconvexURDFLoader
                        loader = NonconvexURDFLoader()
                        loader.set_scene(scene)
                        return loader
                    urdf_builder = create_nonconvex_urdf_loader(self.scene)
                    urdf_builder.fix_root_link = True
                    urdf_builder.load_multiple_collisions_from_file = False
                    if 'scale' in object_cfg:
                        urdf_builder.scale = object_cfg['scale']
                    if 'density' in object_cfg:
                        urdf_builder._density = object_cfg['density']
                    setattr(self.env, object_name, urdf_builder.load(object_file_path))
                else:
                    raise ValueError
                self.movable_objects[object_name] = getattr(self.env, object_name)



def get_scene_builder():
    return TableSceneBuilder