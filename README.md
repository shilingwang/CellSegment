# CellSegment
This is a tool to generate a voronoi diagram representation of cell segmentation. 

After compiling the code, you can input the cell image.
In the folder "./data" you can find a sample image.
The image should be a cell image with red channel representing the cell membrane and blue channel representing the cell cores.

The default assumption is that cell cores are the center of cell shape, which most of the time is not true. 
Therefore, key "Tab" can change the selected cell center.
You can use the keys 'w' for up, 's' for down, 'a' for left, 'd' for right to tune the position of the seleted center.
By tuning the selected center, the algorithm compute new voronoi diagram.
The track bar can change the width of the diagram edge, which is used for the optimization algorithm.

Key "q" will kill the program
