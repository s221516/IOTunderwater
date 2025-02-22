
startMenu = ["Test connection", "Change frequency of carrier wave", "Exit"]

def display_menu(menu_items):

    while True:
        # Display the menu
        print("Please choose an option:")
        for i, item in enumerate(menu_items, start=1):
            print(f"{i}. {item}")

        # Get user input
        try:
            choice = int(input("\n Enter the number of your choice: "))

            if 1 <= choice <= len(menu_items):
                return menu_items[choice - 1]  # Return the selected menu item
            else:
                print("Invalid choice. Please select a valid option.\n")

        except ValueError:
            print("Invalid input. Please enter a number.")
