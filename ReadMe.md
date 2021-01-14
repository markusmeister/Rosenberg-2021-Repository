# Maze Analysis Repository

MM 1/14/2021

This repo accompanies our preprint

**Matthew Rosenberg, Tony Zhang, Pietro Perona, Markus Meister (2021) Mice in a labyrinth: Rapid learning, sudden insight, and efficient exploration**

It contains all the data and code needed to reproduce the published analysis.

## Contents of the repo
`Maze_Analysis_3A`,...,`Maze_Analysis_3D`. These four jupyter notebooks gradually develop the various topics of analysis, starting from raw data, producing figure panels and numerical results for the article along the way. They contain a good number of comments and mathematical sections to guide the user.

`code/`: Contains python files with routines accessed from multiple notebooks.

`outdata/`: A place for data files, both input and output. 

`outdata - tf files only`: Just the raw data, the starting point for all analysis.

`figs/`: A place for PDF files that make up the figure panels in the article.

## How to reproduce all the analysis starting from raw data

0. Read our paper. A version of Jan 2021 is included in the repo. Then read at least the start of `Maze_Analysis_3A`.  
1. Empty the `outdata/` directory. Fill it with the contents of `outdata - tf files only`. Now you're starting with the raw trajectories of animals in the maze.
2. Empty the `figs/` directory.
3. Run the four notebooks `Maze_Analysis_3A`,...,`Maze_Analysis_3D` in alphabetical sequence.
4. Now the `figs/` directory should contain all the figure panels plus a few extras. 

## How to find code for a specific figure panel
- The names of all the figure panels (as numbered in the preprint of Jan 2021) appear as level-3 headings in the notebooks. Look through these to find your figure of interest. Or...
- In the `figs/` directory find the name of the PDF file of interest, and search for that name in the notebooks.

## How to view the raw videos
You can find these on Youtube:

- [Rewarded animals](https://www.youtube.com/playlist?list=PLm5UsX091_2X0ph_ldO3_lC9KFxqYpqo5)
- [Unrewarded animals](https://www.youtube.com/playlist?list=PLm5UsX091_2VTPPMrEEkTsFT8xbFdNi9I)