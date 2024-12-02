import os

def log_model_structure(model, folder_name, filename="model_structure.txt"):
    """
    Prints the structure of the given model, including each layer's input and output shapes.
    If a layer has scale weights and biases, their shapes are also printed.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)

    with open(file_path, 'w') as f:
        # Redirect print output to the file
        original_stdout = os.sys.stdout
        os.sys.stdout = f

        print("\n" + "-" * 80)  # Add lines of dashes before

        print("MODEL STRUCTURE")

        for i, layer in enumerate(model.layers):
            print(f"\nLAYER {i}: {layer}")

#            attributes = dir(layer)
#            for attr in attributes:
#                try:
                    # Print the attribute and its value
#                    print(f"  - {attr}: {getattr(layer, attr)}")
#                except Exception as e:
#                    print(f"  - {attr}: (Could not retrieve: {str(e)})")


            print(f"  - Input Shape: {layer.input}")
            print(f"  - Output Shape: {layer.output}")
            if hasattr(layer, 'get_scale_w') and layer.get_scale_w() is not None:
                print(f"  - Scale Shape of w: {layer.get_scale_w().shape}")
            if hasattr(layer, 'get_scale_b') and layer.get_scale_b() is not None:
                print(f"  - Scale Shape of b: {layer.get_scale_b().shape}")

        print("-" * 80)  # Add lines of dashes after

        # Restore original stdout
        os.sys.stdout = original_stdout

    # Read the file and print its contents to the terminal
    #with open(file_path, 'r') as f:
    #    file_contents = f.read()
    #    print(file_contents)
