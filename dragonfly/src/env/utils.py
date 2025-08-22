import os
import ast
import importlib.util

def find_class_in_folder(folder_path: str, class_name: str):
    """
    Searches for a class definition with the given name in all .py files within folder_path

    Args:
        folder_path: path to the folder containing Python files
        class_name: name of the class to search for

    Returns:
        matching_files: list of file paths where the class is defined
    """
    matching_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):  # Look only for Python files
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=file_path)

                    # Check for class definitions in the parsed AST
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            matching_files.append(file_path)
                            break  # No need to check further in this file

                except (SyntaxError, UnicodeDecodeError) as e:
                    print(f"Skipping {file_path} due to error: {e}")

    return matching_files

def import_class_from_file(file_path: str, class_name: str):
    """
    Dynamically imports a class from a given Python file path

    Args:
        file_path: absolute or relative path to the Python file
        class_name: name of the class to import

    Returns:
        the class object if found, else None
    """
    if not os.path.isfile(file_path) or not file_path.endswith(".py"):
        raise ValueError(f"Invalid Python file: {file_path}")

    module_name = os.path.splitext(os.path.basename(file_path))[
        0
    ]  # Extract module name

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # Load the module

        # Retrieve the class from the module
        if hasattr(module, class_name):
            return getattr(module, class_name)
        else:
            raise ImportError(f"Class '{class_name}' not found in '{file_path}'")

    raise ImportError(f"Could not load module from '{file_path}'")
