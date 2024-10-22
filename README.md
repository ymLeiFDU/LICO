# LICO: Explainable Models with Language-Image COnsistency (NeurIPS 2023)

## Abstract & Framework
Framework of LICO is shown in the following figure:
![schematic](figures/framework.jpg)

## Results
Qualitative and quantitative results obtained by baselines and their LICO versions.
![schematic](figures/cams_and_curves.jpg)

## Response to Reproducibility Study at TMLR
We recently read a reproduction study of our paper and found that the team encountered challenges in reproducing our results, which led to contrasting conclusions. We have carefully read their paper and code and **found coding errors and misunderstandings of our methods.** Specifically, this includes erroneous implementation of Text Features and data processing, **which was inconsistent with our methods, such as incorrect dimensions of text features and prompt tokens.** In addition, we found some other settings that affected the results**. Detailed information can be found in this attachment: "Response to Reproducibility of LICO.pdf"

We hope our report can help in addressing these issues and **correcting erroneous conclusions.** We also apologize for not having updated our code in time and we have now updated our code.

## References

If you find the code useful for your research, please consider citing
```bib
@inproceedings{lei2023lico,
  title={LICO: Explainable Models with Language-Image COnsistency},
  author={Lei, Yiming and Li, Zilong and Li, Yangyang and Zhang, Junping and Shan, Hongming},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


