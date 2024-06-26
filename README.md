<h1 align="center">In-Context Matting [CVPR 2024, Highlight]</h1>


<p align="center">
<a href="https://arxiv.org/pdf/2403.15789.pdf"><img  src="demo/src/icon/arXiv-Paper.svg" ></a>
<!-- <a href="https://link.springer.com/article/"><img  src="demo/src/icon/publication-Paper.svg" ></a> -->
<a href="https://opensource.org/licenses/MIT"><img  src="demo/src/icon/license-MIT.svg"></a>

</p>


<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2403.15789">In-Context Matting</a>.</h4>

<h4 align="center">Details of the model architecture and experimental results can be found in <a href="https://tiny-smart.github.io/icm.github.io/">our homepage</a>.</h4>

## TODO:
- [x] Release code
- [x] Release pre-trained models and instructions for inference
- [x] Release ICM-57 dataset
- [ ] Release training dataset and instructions for training

## Requirements
We follow the environment setup of [Stable Diffusion Version 2](https://github.com/Stability-AI/StableDiffusion#requirements).

## Usage

To evaluate the performance on the ICM-57 dataset using the `eval.py` script, follow these instructions:

1. **Download the Pretrained Model:**
   - Download the pretrained model from [this link](https://pan.baidu.com/s/1HPbRRE5ZtPRpOSocm9qOmA?pwd=BA1c).

2. **Prepare the dataset:**
   Ensure that your ICM-57 is ready.

3. **Run the Evaluation:**
   Use the following command to run the evaluation script. Replace the placeholders with the actual paths if they differ.

   ```bash
   python eval.py --checkpoint PATH_TO_MODEL --save_path results/ --config config/eval.yaml
   ```

### Dataset
**ICM-57**
- Download link: [ICM-57 Dataset](https://pan.baidu.com/s/1ZJU_XHEVhIaVzGFPK_XCRg?pwd=BA1c)
- **Installation Guide**:
  1. After downloading, unzip the dataset into the `datasets/` directory of the project.
  2. Ensure the structure of the dataset folder is as follows:
     ```
     datasets/ICM57/
     ├── image
     └── alpha
     ```

### Acknowledgments

We would like to express our gratitude to the developers and contributors of the [DIFT](https://github.com/Tsingularity/dift) and [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) projects. Their shared resources and insights have significantly aided the development of our work.

## Statement

<!-- If you are interested in our work, please consider citing the following:
```

``` -->

This project is under the MIT license. For technical questions, please contact <strong><i>He Guo</i></strong> at [hguo01@hust.edu.cn](mailto:hguo01@hust.edu.cn). For commerial use, please contact <strong><i>Hao Lu</i></strong> at [hlu@hust.edu.cn](mailto:hlu@hust.edu.cn)
