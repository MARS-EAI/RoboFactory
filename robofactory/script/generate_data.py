import argparse
import os
from robofactory import CONFIG_DIR

TASK_FILE_MAP = {
    'CameraAlignment-rf': 'camera_alignment.yaml',
    'LiftBarrier-rf': 'lift_barrier.yaml',
    'LongPipelineDelivery-rf': 'long_pipeline_delivery.yaml',
    'PassShoe-rf': 'pass_shoe.yaml',
    'PickMeat-rf': 'pick_meat.yaml',
    'PlaceFood-rf': 'place_food.yaml',
    'StackCube-rf': 'stack_cube.yaml',
    'StrikeCube-rf': 'strike_cube.yaml',
    'TakePhoto-rf': 'take_photo.yaml',
    'ThreeRobotsStackCube-rf': 'three_robots_stack_cube.yaml',
    'TwoRobotsStackCube-rf': 'two_robots_stack_cube.yaml',
}

SCENE_TASK_SUPPORT = {
    'robocasa': ['CameraAlignment-rf', 'LiftBarrier-rf', 'LongPipelineDelivery-rf', 'PassShoe-rf', 'PickMeat-rf', 'PlaceFood-rf', 'StackCube-rf', 'StrikeCube-rf','TakePhoto-rf','ThreeRobotsStackCube-rf','TwoRobotsStackCube-rf',],
    'table': ['CameraAlignment-rf', 'LiftBarrier-rf', 'LongPipelineDelivery-rf', 'PassShoe-rf', 'PickMeat-rf', 'PlaceFood-rf', 'StackCube-rf', 'StrikeCube-rf','TakePhoto-rf','ThreeRobotsStackCube-rf','TwoRobotsStackCube-rf',],
}

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner to generate data.")
    parser.add_argument('--config', type=str, default=None, help="Task config file to use")
    parser.add_argument('--scene', type=str, help="Task config file to use", default=None)
    parser.add_argument('--task', type=str, help="Task config file to use", default=None)
    parser.add_argument('--num', type=int, help="Number of trajectories to generate.", required=True)
    parser.add_argument('--save-video', action='store_true', help="Save video of the generated trajectories.")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    args = parser.parse_args()
    
    if (args.config, args.scene, args.task) == (None, None, None):
        raise ValueError('Please give a config path or give scene and task')
    elif args.config is not None and (args.scene, args.task) == (None, None,):
        config_path = args.config
    elif args.config is None:
        if args.scene is not None and args.task is not None:
            if args.task not in TASK_FILE_MAP.keys():
                raise ValueError(f'Unsupport task {args.task}. Now support tasks {list(TASK_FILE_MAP.keys())}')
            if args.scene not in SCENE_TASK_SUPPORT.keys():
                raise ValueError(f'Unsupport scene {args.scene}. Now support scene {list(SCENE_TASK_SUPPORT.keys())}')
            config_path = os.path.join(CONFIG_DIR, args.scene, TASK_FILE_MAP[args.task])
        else:
            raise ValueError('Please give the value both scene and task')

    command = (
        f"python -m robofactory.planner.run "
        f"-c \"{config_path}\" " 
        f"-o=\"rgb\" "
        f"--render-mode=\"sensors\" "
        f"-b=\"cpu\" "
        f"-n {args.num} "
        f"--only-count-success "
        f"--record-dir {args.record_dir} "
        + (f"--save-video" if args.save_video else "")
    )
    print("command: ", command)
    os.system(command)

if __name__ == "__main__":
    main()
