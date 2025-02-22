from datetime import datetime, timedelta
import os

def generate_weekly_markdown(week_offset=0):
    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday()) + timedelta(weeks=week_offset)  # Adjust based on user input
    month_folder = start_of_week.strftime('%b%y')  # Format as Jan25, Feb25, etc.
    base_filename = "week_{}_{}_{}".format(start_of_week.day, start_of_week.month, start_of_week.year)
    relative_folder = os.path.join("Planning & group contract", "daily notes", month_folder)
    os.makedirs(relative_folder, exist_ok=True)  # Ensure directory exists
    
    # Avoid overwriting existing files
    file_counter = 1
    file_path = os.path.join(relative_folder, base_filename + ".md")
    while os.path.exists(file_path):
        file_path = os.path.join(relative_folder, f"{base_filename}_{file_counter}.md")
        file_counter += 1
    
    markdown = "# Task Checklist (Week of {}/{}/{})\n\n".format(start_of_week.day, start_of_week.month, start_of_week.year)
    
    for i in range(7):  # Loop through the 7 days of the week
        day = start_of_week + timedelta(days=i)
        markdown += "- [ ] **{} ({}/{}/{})**\n".format(day.strftime('%A'), day.day, day.month, day.year)
        markdown += "  - [ ] .\n  - [ ] .\n  - [ ] .\n\n"
    
    return file_path, markdown

if __name__ == "__main__":
    week_offset = int(input("Enter week offset (0 for current week, -1 for last week, 1 for next week, etc.): "))
    file_path, markdown_output = generate_weekly_markdown(week_offset)
    with open(file_path, "w") as file:
        file.write(markdown_output)
    print(f"Markdown checklist generated: {file_path}")
