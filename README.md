# graph-rcnn.pytorch (forked)

## What I did
- fix out of memory(OOM) error during inference
  - remove result dictionary (result_dict keeps all results of inference for each iteration)
  - update metric score for each interation
- implement scene graph metric ('sgcls', 'predcls')
  - use gt boxes for proposals
- print triples only for --visualize

## Benchmarking

### Object Detection

source  | backbone | model | bs | lr  | lr_decay | mAP@0.5 | mAP@0.50:0.95
--------|--------|--------|:------:|:------:|:-------:|:------:|:------:
[this repo](https://drive.google.com/open?id=1THLvK8q2VRx6K3G7BGo0FCe-D0EWP9o1) | Res-101 | faster r-cnn | 6 | 5e-3 | 70k,90k | 24.8 | 12.8

### Scene Graph Generation (Frequency Prior Only)
source | backbone | model | bs | lr | lr_decay | sgdet@20 | sgdet@50 | sgdet@100
-------|--------|--------|:------:|:-------:|:------:|:------:|:-------:|:-------:
[this repo](https://drive.google.com/open?id=1Vb-gX3_OLhzgdNseXgS_2DiLmJ8qiG8P) | Res-101 | freq | 6 | 5e-3 | 70k,90k | 19.4 | 25.0 | 28.5
[motifnet](https://github.com/rowanz/neural-motifs) | VGG-16 | freq | - | - | - | 17.7 | 23.5 | 27.6
<!-- Resnet-101 | freq-overlap | 6 | 5e-3 | (70k, 90k) | 100k | - | - | - -->
\* freq = frequency prior baseline

### Scene Graph Generation (Joint training)
source | backbone | model | bs | lr | lr_decay | sgdet@20 | sgdet@50 | sgdet@100
-------|--------|--------|:------:|:-------:|:------:|:------:|:-------:|:-------:
[this repo](https://drive.google.com/open?id=1Vb-gX3_OLhzgdNseXgS_2DiLmJ8qiG8P) | Res-101 | vanilla | 6 | 5e-3 | 70k,90k | 10.4 | 14.3 | 16.8
<!---[this repo](https://drive.google.com/open?id=1Vb-gX3_OLhzgdNseXgS_2DiLmJ8qiG8P) | Res-101 | freq | 6 | 5e-3 | 70k,90k | 100k | 19.4 | 25.0 | 28.5-->

### Scene Graph Generation (Step training)
source | backbone | model | bs | lr | lr_decay | iter | mAP@0.5 | sgdet@20 | sgdet@50 | sgdet@100
-------|--------|--------|:------:|:-------:|:------:|:------:|:-------:|:-------:|:-------:|:-------:
this repo | Res-101 | vanilla | 8 | 5e-3 | 20k,30k | - | 24.8 | 10.5 | 13.8 | 16.1
this repo | Res-101 | imp | 8 | 5e-3 | 20k,30k | - | 24.2 |16.7 | 21.7 | 25.2
[motifnet](https://github.com/rowanz/neural-motifs) | VGG-16 | imp | -| - | - | - | - | 14.6 | 20.7 | 24.5
this repo | Res-101 | msdn | 8 | 5e-3 | 20k,30k | - | - | - | - | -
this repo | Res-101 | grcnn | 8 | 5e-3 | 20k,30k | - | - | - | - | -
my_repo | Res-101 | grcnn | 4 | 5e-3 | 8k,12k | 15k | - | 12.9 | 17.8 | 20.8

