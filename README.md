# C<sup>2</sup>MT: A Novel Credible and Class-aware Multi-task Transformer for SRIQA

C<sup>2</sup>MT code for the following paper:

- H. Li, K. Zhang, Z. Niu and H. Shi, "C<sup>2</sup>MT: A Credible and Class-Aware Multi-Task Transformer for SR-IQA," in [IEEE Signal Processing Letters](https://ieeexplore.ieee.org/document/9999341), vol. 29, pp. 2662-2666, 2022, doi: 10.1109/LSP.2022.3232289.

![Framework](Framework.png)

## Description

### Metadata
In this directory, we build the metadata files for the five benckmark SRIQA databases, which are used for the data loading in our program.

Noting that the number of samples in SISAR metadata file is different to the paper because of the file missing in the latest version.

### Model
In this directory, there are necessary utils for this program, like the network and our proposed other modules.

### Model training
You can run this `trainer_CL.py` for model training after necessary changes, like related parameters and the root in your machine (in `config.yaml`).

## Requirement
- PyTorch 1.11.0

All experiments are performed on Intel Xeon Silver 4210R CPU and Nvidia RTX3090 GPU.

Noting that the results may be still not the same among different implement devices. See [randomness@Pytorch Docs](https://pytorch.org/docs/stable/notes/randomness.html).

## Citation
If you find this work is useful for you, please cite the following paper:
>@ARTICLE{9999341,  
>author={Li, Hui and Zhang, Kaibing and Niu, Zhenxing and Shi, Hongyu},  
>journal={IEEE Signal Processing Letters},   
>title={C<sup>2</sup>MT: A Credible and Class-Aware Multi-Task Transformer for SR-IQA},   
>year={2022},  
>volume={29},  
>pages={2662-2666},  
>doi={10.1109/LSP.2022.3232289}}
