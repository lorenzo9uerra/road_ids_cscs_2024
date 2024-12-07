# AI-Driven Intrusion Detection Systems (IDS) on the ROAD dataset: A Comparative Analysis for automotive Controller Area Network (CAN) - CSCS at ACM CCS 2024

[[dl.acm.org]](https://dl.acm.org/doi/10.1145/3689936.3694696)[[arxiv]](https://arxiv.org/abs/2408.17235)

This is a refactored version of the code used to produce part of the results of the paper on IDS for CAN.

## Citation

```
@inproceedings{10.1145/3689936.3694696,
author = {Guerra, Lorenzo and Xu, Linhan and Bellavista, Paolo and Chapuis, Thomas and Duc, Guillaume and Mozharovskyi, Pavlo and Nguyen, Van-Tam},
title = {AI-Driven Intrusion Detection Systems (IDS) on the ROAD Dataset: A Comparative Analysis for Automotive Controller Area Network (CAN)},
year = {2024},
isbn = {9798400712326},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3689936.3694696},
doi = {10.1145/3689936.3694696},
abstract = {The integration of digital devices in modern vehicles has revolutionized automotive technology, enhancing safety and the overall driving experience. The Controller Area Network (CAN) bus is a central system for managing in-vehicle communication between the electronic control units (ECUs). However, the CAN protocol poses security challenges due to inherent vulnerabilities, lacking encryption and authentication, which, combined with an expanding attack surface, necessitates robust security measures. In response to this challenge, numerous Intrusion Detection Systems (IDS) have been developed and deployed. Nonetheless, an open, comprehensive, and realistic dataset to test the effectiveness of such IDSs remains absent in the existing literature. This paper addresses this gap by considering the latest ROAD dataset, containing stealthy and sophisticated injections. The methodology involves dataset labeling and the implementation of both state-of-the-art deep learning models and traditional machine learning models to show the discrepancy in performance between the datasets most commonly used in the literature and the ROAD dataset, a more realistic alternative.},
booktitle = {Proceedings of the 2024 on Cyber Security in CarS Workshop},
pages = {39–49},
numpages = {11},
keywords = {aiot, can, controller area network, ids, intrusion detection system, road dataset},
location = {Salt Lake City, UT, USA},
series = {CSCS '24}
}
```