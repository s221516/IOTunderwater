# idf_component_register(SRCS "main.c"
#                     INCLUDE_DIRS ".")


idf_component_register(
    SRCS "main.c" "transmitter.c"  # Include both main.c and transmitter.c
    INCLUDE_DIRS "."               # Include the current directory for header files
)
