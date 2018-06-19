import ClearMap.IO.TIF as tif

def read_image(filename):
    """
    Uses the ClearMap code to read TIFF stacks.
        Input:
            <string> filename

        Output:
            <int array [3D]> img [pixel intensities]

    """
    img = tif.readData(filename)
    return img