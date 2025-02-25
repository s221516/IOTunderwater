import subprocess

# Run to save git faster :)

def run_command(command):
    """Runs a command in the shell and returns the output."""
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)

def git_save(commit_message):
    """Runs git pull, add, commit, and push."""
    run_command("git pull")  # Always pull the latest changes first
    run_command("git add .")
    run_command(f'git commit -m "{commit_message}"')
    run_command("git push")

if __name__ == "__main__":
    # Prompt user for commit message
    commit_message = input("Enter your commit message: ")

    if commit_message.strip() == "":
        print("Error: Commit message cannot be empty!")
    else:
        git_save(commit_message)
