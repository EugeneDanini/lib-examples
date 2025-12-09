import os
import subprocess


def run():
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(['adk', 'web', agent_dir])


if __name__ == '__main__':
    run()
