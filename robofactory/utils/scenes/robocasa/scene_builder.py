import importlib
import os
import json
import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat
import yaml
from copy import deepcopy
from typing import Dict
import copy

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import (
    Accessory,
    CoffeeMachine,
    Stool,
    Toaster,
    WallAccessory,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    Drawer,
    HingeCabinet,
    HousingCabinet,
    OpenCabinet,
    PanelCabinet,
    SingleCabinet,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.dishwasher import Dishwasher
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import (
    Fixture,
    FixtureType,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture_stack import FixtureStack
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture_utils import (
    fixture_is_type,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.fridge import Fridge
from mani_skill.utils.scene_builder.robocasa.fixtures.hood import Hood
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Box, Floor, Wall
from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Oven, Stove, Stovetop
from mani_skill.utils.scene_builder.robocasa.fixtures.windows import (
    FramedWindow,
    Window,
)
from mani_skill.utils.scene_builder.robocasa.utils import object_utils as OU
from mani_skill.utils.scene_builder.robocasa.utils import scene_registry, scene_utils
from mani_skill.utils.scene_builder.robocasa.utils.placement_samplers import (
    RandomizationError,
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from mani_skill.utils.scene_builder.scene_builder import SceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose

from ..scene_builder import RFSceneBuilder
from ..utils import merge_dicts

FIXTURES = dict(
    hinge_cabinet=HingeCabinet,
    single_cabinet=SingleCabinet,
    open_cabinet=OpenCabinet,
    panel_cabinet=PanelCabinet,
    housing_cabinet=HousingCabinet,
    drawer=Drawer,
    counter=Counter,
    stove=Stove,
    stovetop=Stovetop,
    oven=Oven,
    microwave=Microwave,
    hood=Hood,
    sink=Sink,
    fridge=Fridge,
    dishwasher=Dishwasher,
    wall=Wall,
    floor=Floor,
    box=Box,
    accessory=Accessory,
    paper_towel=Accessory,
    plant=Accessory,
    knife_block=Accessory,
    stool=Stool,
    utensil_holder=Accessory,
    coffee_machine=CoffeeMachine,
    toaster=Toaster,
    utensil_rack=WallAccessory,
    wall_accessory=WallAccessory,
    window=Window,
    framed_window=FramedWindow,
)

# fixtures that are attached to other fixtures, disables positioning system in this script
FIXTURES_INTERIOR = dict(
    sink=Sink, stovetop=Stovetop, accessory=Accessory, wall_accessory=WallAccessory
)

ALL_SIDES = ["left", "right", "front", "back", "bottom", "top"]

ROBOT_FRONT_FACING_SIZE = dict(fetch=0.8, unitree_g1_simplified_upper_body=0.6)


def check_syntax(fixture):
    """
    Checks that specifications of a fixture follows syntax rules
    """

    if fixture["type"] != "stack" and fixture["type"] not in FIXTURES:
        raise ValueError(
            'Invalid value for fixture type: "{}".'.format(fixture["type"])
        )

    if "config_name" in fixture and "default_config_name" in fixture:
        raise ValueError('Cannot specify both "config_name" and "default_config_name"')

    if "align_to" in fixture or "side" in fixture or "alignment" in fixture:
        if not ("align_to" in fixture and "side" in fixture):
            raise ValueError(
                'Both or neither of "align_to" and ' '"side" need to be specified.'
            )
        if "pos" in fixture:
            raise ValueError("Cannot specify both relative and absolute positions.")

        # check alignment and side arguments are compatible
        if "alignment" in fixture:
            for keywords in scene_utils.AXES_KEYWORDS.values():
                if fixture["side"] in keywords:
                    # check that neither keyword is used for alignment
                    if (
                        keywords[0] in fixture["alignment"]
                        or keywords[1] in fixture["alignment"]
                    ):
                        raise ValueError(
                            'Cannot set alignment to "{}" when aligning to the "{}" side'.format(
                                fixture["alignment"], fixture["side"]
                            )
                        )

        # check if side is valid
        if fixture["side"] not in ALL_SIDES:
            raise ValueError(
                '"{}" is not a valid side for alignment'.format(fixture["side"])
            )



# TODO: build isolated TableSceneBuilder and RoboCasaSceneBuilder
class RobocasaSceneBuilder(RFSceneBuilder):
    def build(self):
        cfg = copy.deepcopy(self.cfg)
        self.scene_data = []  # maps scene_idx to {"fixtures", "fxtr_placements"}
        scene_idx = cfg['scene']['env']['scene_idx']
        layout_idx = cfg['scene']['env']['layout_idx']
        style_idx = cfg['scene']['env']['style_idx']
        layout_path = scene_registry.get_layout_path(layout_idx)
        style_path = scene_registry.get_style_path(style_idx)
        
        # load style
        with open(style_path, "r") as f:
            style = yaml.safe_load(f)
        # load arena
        with open(layout_path, "r") as f:
            arena_config = yaml.safe_load(f)
        if 'customized_style' in cfg['scene']['env']:
            with open(cfg['scene']['env']['customized_style'], "r") as f:
                customized_style = yaml.safe_load(f)
                merge_dicts(style, customized_style)
        if 'customized_layout' in cfg['scene']['env']:
            with open(cfg['scene']['env']['customized_layout'], "r") as f:
                customized_layout = yaml.safe_load(f)
                merge_dicts(arena_config, customized_layout)
        # contains all fixtures with updated configs
        arena = list()

        # Update each fixture config. First iterate through groups: subparts of the arena that can be
        # rotated and displaced together. example: island group, right group, room group, etc
        for group_name, group_config in arena_config.items():
            group_fixtures = list()
            # each group is further divded into similar subcollections of fixtures
            # ex: main group counter accessories, main group top cabinets, etc
            for k, fixture_list in group_config.items():
                # these values are rotations/displacements that are applied to all fixtures in the group
                if k in ["group_origin", "group_z_rot", "group_pos"]:
                    continue
                elif type(fixture_list) != list:
                    raise ValueError(
                        '"{}" is not a valid argument for groups'.format(k)
                    )

                # add suffix to support different groups
                for fxtr_config in fixture_list:
                    fxtr_config["name"] += "_" + group_name
                    # update fixture names for alignment, interior objects, etc.
                    for k in scene_utils.ATTACH_ARGS + [
                        "align_to",
                        "stack_fixtures",
                        "size",
                    ]:
                        if k in fxtr_config:
                            if isinstance(fxtr_config[k], list):
                                for i in range(len(fxtr_config[k])):
                                    if isinstance(fxtr_config[k][i], str):
                                        fxtr_config[k][i] += "_" + group_name
                            else:
                                if isinstance(fxtr_config[k], str):
                                    fxtr_config[k] += "_" + group_name

                group_fixtures.extend(fixture_list)

            # update group rotation/displacement if necessary
            if "group_origin" in group_config:
                for fxtr_config in group_fixtures:
                    # do not update the rotation of the walls/floor
                    if fxtr_config["type"] in ["wall", "floor"]:
                        continue
                    fxtr_config["group_origin"] = group_config["group_origin"]
                    fxtr_config["group_pos"] = group_config["group_pos"]
                    fxtr_config["group_z_rot"] = group_config["group_z_rot"]

            # addto overall fixture list
            arena.extend(group_fixtures)

        # maps each fixture name to its object class
        fixtures: Dict[str, Fixture] = dict()
        # maps each fixture name to its configuration
        configs = dict()
        # names of composites, delete from fixtures before returning
        composites = list()

        for fixture_config in arena:
            # scene_registry.check_syntax(fixture_config)
            fixture_name = fixture_config["name"]

            # stack of fixtures, handled separately
            if fixture_config["type"] == "stack":
                stack = FixtureStack(
                    self.scene,
                    fixture_config,
                    fixtures,
                    configs,
                    style,
                    default_texture=None,
                    rng=self.env._batched_episode_rng[scene_idx],
                )
                fixtures[fixture_name] = stack
                configs[fixture_name] = fixture_config
                composites.append(fixture_name)
                continue

            # load style information and update config to include it
            default_config = scene_utils.load_style_config(style, fixture_config)
            if default_config is not None:
                for k, v in fixture_config.items():
                    default_config[k] = v
                fixture_config = default_config

            # set fixture type
            if fixture_config["type"] not in FIXTURES:
                continue
            fixture_config["type"] = FIXTURES[fixture_config["type"]]

            # pre-processing for fixture size
            size = fixture_config.get("size", None)
            if isinstance(size, list):
                for i in range(len(size)):
                    elem = size[i]
                    if isinstance(elem, str):
                        ref_fxtr = fixtures[elem]
                        size[i] = ref_fxtr.size[i]

            # initialize fixture
            # TODO (stao): use batched episode rng later
            fixture = scene_utils.initialize_fixture(
                self.scene,
                fixture_config,
                fixtures,
                rng=self.env._batched_episode_rng[scene_idx],
            )

            fixtures[fixture_name] = fixture
            configs[fixture_name] = fixture_config
            pos = None
            # update fixture position
            if fixture_config["type"] not in FIXTURES_INTERIOR.values():
                # relative positioning
                if "align_to" in fixture_config:
                    pos = scene_utils.get_relative_position(
                        fixture,
                        fixture_config,
                        fixtures[fixture_config["align_to"]],
                        configs[fixture_config["align_to"]],
                    )

                elif "stack_on" in fixture_config:
                    stack_on = fixtures[fixture_config["stack_on"]]

                    # account for off-centered objects
                    stack_on_center = stack_on.center

                    # infer unspecified axes of position
                    pos = fixture_config["pos"]
                    if pos[0] is None:
                        pos[0] = stack_on.pos[0] + stack_on_center[0]
                    if pos[1] is None:
                        pos[1] = stack_on.pos[1] + stack_on_center[1]

                    # calculate height of fixture
                    pos[2] = (
                        stack_on.pos[2] + stack_on.size[2] / 2 + fixture.size[2] / 2
                    )
                    pos[2] += stack_on_center[2]
                else:
                    # absolute position
                    pos = fixture_config.get("pos", None)
            if pos is not None and type(fixture) not in [Wall, Floor]:
                fixture.set_pos(deepcopy(pos))
        # composites are non-MujocoObjects, must remove
        for composite in composites:
            del fixtures[composite]

        # update the rotation and postion of each fixture based on their group
        for name, fixture in fixtures.items():
            # check if updates are necessary
            config = configs[name]
            if "group_origin" not in config:
                continue

            # TODO: add default for group origin?
            # rotate about this coordinate (around the z-axis)
            origin = config["group_origin"]
            pos = config["group_pos"]
            z_rot = config["group_z_rot"]
            displacement = [pos[0] - origin[0], pos[1] - origin[1]]

            if type(fixture) not in [Wall, Floor]:
                dx = fixture.pos[0] - origin[0]
                dy = fixture.pos[1] - origin[1]
                dx_rot = dx * np.cos(z_rot) - dy * np.sin(z_rot)
                dy_rot = dx * np.sin(z_rot) + dy * np.cos(z_rot)

                x_rot = origin[0] + dx_rot
                y_rot = origin[1] + dy_rot
                z = fixture.pos[2]
                pos_new = [x_rot + displacement[0], y_rot + displacement[1], z]

                # account for previous z-axis rotation
                rot_prev = fixture.euler
                if rot_prev is not None:
                    # TODO: switch to quaternion since euler rotations are ambiguous
                    rot_new = rot_prev
                    rot_new[2] += z_rot
                else:
                    rot_new = [0, 0, z_rot]
                fixture.pos = np.array(pos_new)
                fixture.set_euler(rot_new)

        fixture_cfgs = self.get_fixture_cfgs(fixtures)
        # generate initial poses for objects so that they are spawned in nice places during GPU initialization
        # to be more performant
        fxtr_placements = self._generate_initial_placements(
            fixtures, fixture_cfgs, rng=self.env._batched_episode_rng[scene_idx]
        )
        self.scene_data.append(
            dict(
                fixtures=fixtures,
                fxtr_placements=fxtr_placements,
                fixture_cfgs=fixture_cfgs,
            )
        )

        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in fxtr_placements.values():
            assert isinstance(obj, Fixture)
            obj.pos = obj_pos
            obj.quat = obj_quat

        actors: Dict[str, Actor] = {}

        ### collision handling and optimization ###
        # Generally we aim to ensure all articulations in a stack have the same collision bits so they can't collide with each other
        # and with a range of [22, 30] we can generally ensure adjacent articulations can collide with each other.
        # walls and floors cannot collide with anything. Walls can only collide with the robot. They are assigned bits 22 to 30.
        # mobile base robots have their wheels/non base links assigned bit of 30 to not collide with the floor or walls.
        # the base links can optionally be also assigned a bit of 31 to not collide with walls.

        # fixtures that are not articulated are always static and cannot hit other non-articulated fixtures. This scenario is assigned bit 21.
        actor_bit = 21
        collision_start_bit = 22
        fixture_idx = 0
        stack_collision_bits = dict()
        for stack_index, stack in enumerate(composites):
            stack_collision_bits[stack] = collision_start_bit + stack_index % 9
        for k, v in fixtures.items():
            fixture_idx += 1
            built = v.build(scene_idxs=[scene_idx])
            if built is not None:
                actors[k] = built
                # ensure all rooted articulated objects have collisions ignored with all static objects
                # ensure all articulations in the same stack have the same collision bits, since by definition for robocasa they cannot
                # collide with each other
                if (
                    built.is_articulation
                    and built.articulation.fixed_root_link.all()
                ):
                    collision_bit = collision_start_bit + fixture_idx % 5
                    if "stack" in v.name:
                        for stack_group in stack_collision_bits.keys():
                            if stack_group in v.name:
                                collision_bit = stack_collision_bits[stack_group]
                                break
                    for link in built.articulation.links:
                        link.set_collision_group(
                            group=2, value=0
                        )  # clear all default ignored collisions
                        if link.joint.type[0] == "fixed":
                            link.set_collision_group_bit(
                                group=2, bit_idx=actor_bit, bit=1
                            )
                        link.set_collision_group_bit(
                            group=2, bit_idx=collision_bit, bit=1
                        )

                else:
                    if built.actor.px_body_type == "static":
                        collision_bit = collision_start_bit + fixture_idx % 5
                        if "stack" in v.name:
                            for stack_group in stack_collision_bits.keys():
                                if stack_group in v.name:
                                    collision_bit = stack_collision_bits[
                                        stack_group
                                    ]
                                    break
                        if isinstance(v, Floor):
                            for bit_idx in range(21, 32):
                                built.actor.set_collision_group_bit(
                                    group=2, bit_idx=bit_idx, bit=1
                                )
                        elif isinstance(v, Wall):
                            for bit_idx in range(21, 31):
                                built.actor.set_collision_group_bit(
                                    group=2, bit_idx=bit_idx, bit=1
                                )

                        else:
                            built.actor.set_collision_group_bit(
                                group=2,
                                bit_idx=collision_bit,
                                bit=1,
                            )
                            built.actor.set_collision_group_bit(
                                group=2, bit_idx=actor_bit, bit=1
                            )
        scene_cfg = cfg['scene']
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
        if 'objects' in cfg:
            objects_cfg = cfg['objects']
            self.movable_objects = {}
            for object_cfg in objects_cfg:
                object_file_path = object_cfg['file_path']
                object_type = os.path.splitext(object_file_path)[-1]
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
                    self.movable_objects[object_name] = getattr(self.env, object_name)
                elif object_type in ['.urdf']:
                    # use nonconvex collision
                    def create_nonconvex_urdf_loader(scene):
                        from robofactory.utils.building.nonconvex_urdf_loader import NonconvexURDFLoader
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

    def _generate_initial_placements(
        self, fixtures, fixture_cfgs, rng: np.random.RandomState
    ):
        """Generate and places randomized fixtures and robot(s) into the scene. This code is not parallelized"""
        fxtr_placement_initializer = self._get_placement_initializer(
            fixtures, dict(), fixture_cfgs, z_offset=0.0, rng=rng
        )
        fxtr_placements = None
        for i in range(10):
            try:
                fxtr_placements = fxtr_placement_initializer.sample()
            except RandomizationError:
                # if macros.VERBOSE:
                #     print("Ranomization error in initial placement. Try #{}".format(i))
                continue
            break
        if fxtr_placements is None:
            # if macros.VERBOSE:
            # print("Could not place fixtures.")
            # self._load_model()
            raise RuntimeError("Could not place fixtures.")

        # setup internal references related to fixtures
        # self._setup_kitchen_references()
        return fxtr_placements

    # def initialize(self, env_idx: torch.Tensor, task_name : str = "default"):
    #     b = len(env_idx)
    #     scene_cfg = cfg['scene']

    #     # primitive
    #     if 'primitives' in scene_cfg:
    #         for primitive_cfg in scene_cfg['primitives']:
    #             asset = getattr(self.env, primitive_cfg['name'], None)
    #             if not asset:
    #                 raise AttributeError(f'Attribute "{primitive_cfg["name"]}" not found in SceneBuilder.')
    #             qpos = primitive_cfg['pos']['qpos']
    #             if 'randq_scale' in primitive_cfg:
    #                 qpos = np.array(qpos) + np.array(primitive_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
    #                 qpos = qpos.tolist()
    #             asset.set_pose(Pose.create_from_pq(primitive_cfg['pos']['ppos']['p'], qpos))

    #     # objects
    #     if 'objects' in cfg:
    #         objects_cfg = cfg['objects']
    #         self.movable_objects = {}
    #         for asset_cfg in objects_cfg:
    #             asset = getattr(self.env, asset_cfg['name'], None)
    #             if not asset:
    #                 raise AttributeError(f'Attribute "{asset_cfg["name"]}" not found in SceneBuilder.')
    #             qpos = (asset_cfg['pos']['qpos'])
    #             if 'randq_scale' in asset_cfg:
    #                 qpos = np.array(qpos) + np.array(asset_cfg['pos']['randq_scale'])* np.random.rand((len(qpos)))
    #                 qpos = qpos.tolist()
    #             asset.set_pose(Pose.create_from_pq(asset_cfg['pos']['ppos']['p'], qpos))
    #             self.movable_objects[asset_cfg['name']] = asset
    #     # agents
    #     agents_cfg = cfg['agents']
    #     agent: MultiAgent = self.env.agent
    #     self.articulations = {}
    #     for idx, agent_cfg in enumerate(agents_cfg):
    #         pos_cfg = agent_cfg['pos']
    #         ppos = pos_cfg['ppos']['p']
    #         if 'randp_scale' in pos_cfg:
    #             ppos = np.array(ppos) + np.array(agent_cfg['pos']['randp_scale']) * np.random.rand((len(ppos)))
    #             ppos = ppos.tolist()
    #         ppos = sapien.Pose(ppos, q=euler2quat(*pos_cfg['ppos']['q']))
    #         qpos = np.array((pos_cfg['qpos']))
    #         if 'randq_scale' in pos_cfg:
    #             qpos = np.tile(qpos, (b, 1)) + np.tile(np.array(agent_cfg['pos']['randq_scale']), (b, 1)) * np.random.rand(b, (len(qpos)))
    #         agent.agents[idx].reset(qpos)
    #         agent.agents[idx].robot.set_pose(ppos)
    #         self.articulations[agent_cfg['robot_uid']] = agent.agents[idx]

    def get_fixture_cfgs(self, fixtures):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for (name, fxtr) in fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs
    
    def _is_fxtr_valid(self, fxtr, size):
        """
        checks if counter is valid for object placement by making sure it is large enough

        Args:
            fxtr (Fixture): fixture to check
            size (tuple): minimum size (x,y) that the counter region must be to be valid

        Returns:
            bool: True if fixture is valid, False otherwise
        """
        return True
        for region in fxtr.get_reset_regions(self).values():
            if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
                return True
        return False

    def get_fixture(self, fixtures, id, ref=None, size=(0.2, 0.2)):
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

        Returns:
            Fixture: fixture object
        """
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in fixtures.keys():
            return fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                matches = [name for name in fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if self._is_fxtr_valid(fixtures[name], size)
                ]
            assert len(matches) > 0
            # sample random key
            # TODO (stao): fix the key!
            key = self.env._episode_rng.choice(matches)
            return fixtures[key]
        else:
            ref_fixture = self.get_fixture(fixtures, ref)

            assert isinstance(id, FixtureType)
            cand_fixtures = []
            for fxtr in fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = self._is_fxtr_valid(fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def _get_placement_initializer(
        self,
        fixtures,
        objects,
        cfg_list,
        z_offset=0.01,
        rng: np.random.RandomState = None,
    ):

        """
        Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list

        Args:
            cfg_list (list): list of object configurations

            z_offset (float): offset in z direction

        Returns:
            SequentialCompositeSampler: placement initializer

        """

        placement_initializer = SequentialCompositeSampler(name="SceneSampler", rng=rng)

        for (obj_i, cfg) in enumerate(cfg_list):
            # determine which object is being placed
            if cfg["type"] == "fixture":
                mj_obj = fixtures[cfg["name"]]
            elif cfg["type"] == "object":
                mj_obj = objects[cfg["name"]]
            else:
                raise ValueError
            placement = cfg.get("placement", None)
            if placement is None:
                continue
            fixture_id = placement.get("fixture", None)
            if fixture_id is not None:
                # get fixture to place object on
                fixture = self.get_fixture(
                    fixtures,
                    id=fixture_id,
                    ref=placement.get("ref", None),
                )

                # calculate the total available space where object could be placed
                sample_region_kwargs = placement.get("sample_region_kwargs", {})

                reset_region = fixture.sample_reset_region(
                    env=self, fixtures=fixtures, **sample_region_kwargs
                )
                outer_size = reset_region["size"]
                margin = placement.get("margin", 0.04)
                outer_size = (outer_size[0] - margin, outer_size[1] - margin)

                # calculate the size of the inner region where object will actually be placed
                target_size = placement.get("size", None)
                if target_size is not None:
                    target_size = deepcopy(list(target_size))
                    for size_dim in [0, 1]:
                        if target_size[size_dim] == "obj":
                            target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                        if target_size[size_dim] == "obj.x":
                            target_size[size_dim] = mj_obj.size[0] + 0.005
                        if target_size[size_dim] == "obj.y":
                            target_size[size_dim] = mj_obj.size[1] + 0.005
                    inner_size = np.min((outer_size, target_size), axis=0)
                else:
                    inner_size = outer_size

                inner_xpos, inner_ypos = placement.get("pos", (None, None))
                offset = placement.get("offset", (0.0, 0.0))

                # center inner region within outer region
                if inner_xpos == "ref":
                    # compute optimal placement of inner region to match up with the reference fixture
                    x_halfsize = outer_size[0] / 2 - inner_size[0] / 2
                    if x_halfsize == 0.0:
                        inner_xpos = 0.0
                    else:
                        ref_fixture = self.get_fixture(
                            fixtures, placement["sample_region_kwargs"]["ref"]
                        )
                        ref_pos = ref_fixture.pos
                        fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                        outer_to_ref = fixture_to_ref - reset_region["offset"]
                        inner_xpos = outer_to_ref[0] / x_halfsize
                        inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
                elif inner_xpos is None:
                    inner_xpos = 0.0

                if inner_ypos is None:
                    inner_ypos = 0.0
                # offset for inner region
                intra_offset = (
                    (outer_size[0] / 2 - inner_size[0] / 2) * inner_xpos + offset[0],
                    (outer_size[1] / 2 - inner_size[1] / 2) * inner_ypos + offset[1],
                )
                # center surface point of entire region
                ref_pos = fixture.pos + [0, 0, reset_region["offset"][2]]
                ref_rot = fixture.rot

                # x, y, and rotational ranges for randomization
                x_range = (
                    np.array([-inner_size[0] / 2, inner_size[0] / 2])
                    + reset_region["offset"][0]
                    + intra_offset[0]
                )
                y_range = (
                    np.array([-inner_size[1] / 2, inner_size[1] / 2])
                    + reset_region["offset"][1]
                    + intra_offset[1]
                )
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
            else:
                target_size = placement.get("size", None)
                x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
                y_range = np.array([-target_size[1] / 2, target_size[1] / 2])
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
                ref_pos = [0, 0, 0]
                ref_rot = 0.0

            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="{}_Sampler".format(cfg["name"]),
                    mujoco_objects=mj_obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation,
                    ensure_object_boundary_in_range=placement.get(
                        "ensure_object_boundary_in_range", True
                    ),
                    ensure_valid_placement=placement.get(
                        "ensure_valid_placement", True
                    ),
                    reference_pos=ref_pos,
                    reference_rot=ref_rot,
                    z_offset=z_offset,
                    rng=rng,
                    rotation_axis=placement.get("rotation_axis", "z"),
                ),
                sample_args=placement.get("sample_args", None),
            )
        return placement_initializer
    

def get_scene_builder():
    return RobocasaSceneBuilder