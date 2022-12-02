# Pseudo Support: A Sharp Enhancement Method for Few-shot Classification under Extreme Scarcity of Support Images



## Introduction
In this repo, we provide the implementation of the following paper:<br>
"Pseudo Support: A Sharp Enhancement Method for Few-shot Classification under Extreme Scarcity of Support Images"  [[Paper]]().

 When there is a lack of support images, the embeddings extracted from the support images are likely to deviate from the average feature of categories. Consequently, current few-shot classification algorithms work less effectively when support images are extreme scarce as in the 1-shot setting. To alleviate this problem, we propose Pseudo Support, a sharp prototype embedding enhancement method for few-shot classification. In the first classification step, we pick query images of high confidence as pseudo support images, whose feature maps are concatenated to original support feature maps. It is proved with mathematical derivation that prototype embeddings calculated with updated support feature maps are more likely to be close to the average feature of categories. Experiments conducted on four standard few- shot classification benchmarks show that, Pseudo Support boosts the performance of different base models by more than 5%, as well as setting new state-of-the-art results. Further, within an error range of 3%, our method helps a model that learns from 2 images achieve an accuracy close to one that learns from 5 images.


## Few-shot classification Results
Experimental results on miniImageNet tieredImagenet and CUB and CIFAR-FS. The network structures and training settings of ProtoNet and MetaDBC are consistent with the MetaBDC paper "Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification"[https://arxiv.org/pdf/2204.04567.pdf]. Experiments conducted with ReNet follow its original settings in "Relational embedding for few-shot classification.". Our networks are trained on an NVIDIA GeForce GTX 1080ti GPU.
### miniImageNet & tieredImagenet
<table>
         <tr>
             <th rowspan="2" style="text-align:center;">Dataset</th>
             <th colspan="2" style="text-align:center;">miniImageNet</th>
             <th colspan="2" style="text-align:center;">tieredImagenet</th>
         </tr>
         <tr>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">62.11±0.44</td>
             <td style="text-align:center;">80.77±0.30</td>
             <td style="text-align:center;">68.31±0.51</td>
             <td style="text-align:center;">83.85±0.36</td>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet+PS</td>
             <td style="text-align:center;">67.50±0.50</td>
             <td style="text-align:center;">79.87±0.32</td>
             <td style="text-align:center;">71.06±0.56</td>
             <td style="text-align:center;">82.55±0.37</td>
         </tr>
         <tr>
             <td style="text-align:center">MetaBDC</td>
             <td style="text-align:center;">67.34±0.43</td>
             <td style="text-align:center;">84.46±0.28</td>
             <td style="text-align:center;">72.34±0.49</td>
             <td style="text-align:center;">87.31±0.32</td>
         </tr>
         <tr>
             <td style="text-align:center">MeatBDC+PS</td>
             <td style="text-align:center;">70.93±0.50</td>
             <td style="text-align:center;">84.07±0.30</td>
             <td style="text-align:center;">78.50±0.50</td>
             <td style="text-align:center;">86.93±0.33</td>
         </tr>
         <tr>
             <td style="text-align:center">ReNet</td>
             <td style="text-align:center;">67.60±0.44</td>
             <td style="text-align:center;">82.58±0.30</td>
             <td style="text-align:center;">71.61±0.51</td>
             <td style="text-align:center;">85.28±0.35</td>
         </tr>
         <tr>
             <td style="text-align:center">ReNet+PS</td>
             <td style="text-align:center;">72.30±0.50</td>
             <td style="text-align:center;">83.93±0.30</td>
             <td style="text-align:center;">77.20±0.53</td>
             <td style="text-align:center;">86.27±0.34</td>
         </tr>
</table>


### CUB & CIFAR-FS
<table>
         <tr>
             <th rowspan="2" style="text-align:center;">Dataset</th>
             <th colspan="2" style="text-align:center;">CUB</th>
             <th colspan="2" style="text-align:center;">CIFAR-FS</th>
         </tr>
         <tr>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
             <th colspan="1" style="text-align:center;">5-way-1-shot</th>
             <th colspan="1" style="text-align:center;">5-way-5-shot</th>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet</td>
             <td style="text-align:center;">80.90±0.43</td>
             <td style="text-align:center;">89.81±0.23</td>
             <td style="text-align:center;">/</td>
             <td style="text-align:center;">/</td>
         </tr>
         <tr>
             <td style="text-align:center">ProtoNet+PS</td>
             <td style="text-align:center;">83.48±0.44</td>
             <td style="text-align:center;">89.85±0.23</td>
             <td style="text-align:center;">/</td>
             <td style="text-align:center;">/</td>
         </tr>
         <tr>
             <td style="text-align:center">MetaBDC</td>
             <td style="text-align:center;">83.55±0.40</td>
             <td style="text-align:center;">93.82±0.17</td>
             <td style="text-align:center;">/</td>
             <td style="text-align:center;">/</td>
         </tr>
         <tr>
             <td style="text-align:center">MetaBDC+PS</td>
             <td style="text-align:center;">88.35±0.38</td>
             <td style="text-align:center;">93.91±0.16</td>
             <td style="text-align:center;">/</td>
             <td style="text-align:center;">/</td>
         </tr>
         <tr>
             <td style="text-align:center">ReNet</td>
             <td style="text-align:center;">79.49±0.44</td>
             <td style="text-align:center;">91.11±0.24</td>
             <td style="text-align:center;">74.51±0.46</td>
             <td style="text-align:center;">86.60±0.32</td>
         </tr>
         <tr>
             <td style="text-align:center">ReNet+PS</td>
             <td style="text-align:center;">84.00±0.44<</td>
             <td style="text-align:center;">91.43±0.24</td>
             <td style="text-align:center;">80.14±0.48</td>
             <td style="text-align:center;">87.98±0.31</td>
         </tr>
</table>

## References
[DEEPBDC] Jiangtao Xie, Fei Long, Jiaming Lv, Qilong Wang, and Peihua Li. Joint distribution matters: deep brownian distance covariance for few-shot classification. In CVPR, 2022. <br>
[ProtoNet] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In NIPS, 2017.<br>
[ReNet] Dahyun Kang, Heeseung Kwon, Juhong Min, and Minsu Cho. Relational embedding for few-shot classification. In ICCV, 2021.<br> 
Our code is modified on MetaBDC, ProtoNet[https://github.com/Fei-Long121/DeepBDC] and ReNet[https://github.com/dahyun-kang/renet]. 


### **For MetaBDC and ProtoNet on datasets**
#### MetaBDC
1. `cd DEEPBDC-new/scripts/mini_magenet/run_meta_deepbdc_c_8`
2.  modify the dataset path in `run_pretrain.sh`, `run_metatrain.sh` and `run_test.sh`, modify the --fake p to gain the result with the number of pseudo support.
3. `bash run.sh`
#### ProtoNet
1. `cd DEEPBDC-new/scripts/mini_magenet/run_protonet_c`
2.  modify the dataset path in `run_pretrain.sh`, `run_metatrain.sh` and `run_test.sh`, modify the --fake p to gain the result with the number of pseudo support.
3. `bash run.sh`

run_protonet_c_-M+N modify the --fake M --true N to gain the selected N correct and M incorrect images are denoted as -M+N-pseudo.

### **For ReNet on datasets**
#### ReNet
1. `cd renet-new`
`bash scripts/train/dataset_5w1s_c.sh` to gain the ReNet model with 5 way 1 shot added 4 pseudo support.
`dataset belongs to miniimagenet, tieredimagenet, cub, cifar_fs`



## Contact
The current code format is not standardized enough. It is only for reference. If you have any questions, please contact us to modify and discuss them. We will upload the standardized code as soon as possible. If you have any questions or suggestions, please contact us:
`Chen Xu(chenxu20@mails.tsinghua.edu.cn)`<br>





