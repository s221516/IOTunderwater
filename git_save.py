import subprocess
import sys

def run_command(command):
    """Runs a command in the shell and returns the output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)

def git_save(commit_message):
    """Runs git add, commit, and push."""
    run_command("git add .")
    run_command(f'git commit -m "{commit_message}"')
    run_command("git push")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: git_save.py \"Your commit message\"")
    else:
        commit_message = sys.argv[1]
        git_save(commit_message)
