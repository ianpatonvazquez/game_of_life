# Predicting Conway's Game of Life with Neural Networks

This repository contains the code and methods used for the project **"Machine learning models to predict the evolution of Conway’s Game of Life under different levels of coarse-grainings of the system"** conducted during an internship at [ICFO](https://www.icfo.eu/) under the supervision of Dr. Tamás Kriváchy. The project explores how different neural network architectures perform when tasked with predicting the behavior of the Game of Life, especially under spatial coarse-graining.

## Overview

Conway’s Game of Life is a well-known cellular automaton where complex patterns emerge from simple local rules. The aim of this project is to evaluate the ability of machine learning models (particularly **Multilayer Perceptrons (MLPs)** and a bit of **Convolutional Neural Networks (CNNs)**) to predict the state of the center cell after a number of generations, using as input coarse-grained grids evolved over time.

## Aspects covered
- Implementation of a **Game of Life simulator** with periodic boundary conditions.
- Two types of coarse-graining methods: **block-average** and **pseudo-convolutional**.
- Generation of supervised learning datasets with customizable parameters.
- Training and evaluation of MLP and CNN models using **PyTorch**.
- Analysis of performance across different coarse-graining levels and dataset configurations.

## Repository structure
- data: Scripts for generating datasets
- models: MLP and CNN architectures
- train: Training and validation 
- tests: Testing and performance evaluation
- utils: Helper functions
- gifs: Game of Life visualizations
- notebooks: Jupyter notebooks
- requirements.txt: Python dependencies
- main.py: Example script to run the process
- README.md: Project overview
- report.pdf: Final internship report

