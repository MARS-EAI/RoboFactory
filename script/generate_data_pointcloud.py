import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run RoboFactory planner to generate data.")
    parser.add_argument('config', type=str, help="Task config file to use")
    parser.add_argument('num', type=int, help="Number of trajectories to generate.")
    parser.add_argument('--save-video', action='store_true', help="Save video of the generated trajectories.")
    args = parser.parse_args()

    command = (
        f"python -m planner.run "
        f"-c \"{args.config}\" " 
        f"-o=\"pointcloud\" "
        f"--render-mode=\"sensors\" "
        f"-b=\"cpu\" "
        f"-n {args.num} "
        f"--only-count-success "
        + (f"--save-video" if args.save_video else "")
    )
    print("command: ", command)
    os.system(command)

if __name__ == "__main__":
    main()
