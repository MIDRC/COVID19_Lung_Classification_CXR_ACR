# MIDRC CRP 5 COVID19 Lung Classfication CXR ACR

**Development Team**: Laura Brink, Kendall Schmidt

Problem Definition: Classify Chest X-rays as either COVID positive or negative.

**Modality**: X-ray

**Requirements**: See Dockerfile

We began with a torchvision resnet50 algorithm and trained on the labeled CXR MIDRC data. The code shows how to run inference using our new model. We package our model according to AI-LAB's model standards. 

**Repo Content Description**: 
- Dockerfile: includes all package requirements as well as specifies entrypoints.
- inference.py: main; use environment variables to specify the weights file, gpu index. 
- data_preparation.py: called by inference.py

**Example Commands**: 
- docker build -t {imagename} {pathtoDockerfile}
- docker run {imagename}
- Use environment variables to specify the weights file and gpu.
- Use docker volumes to mount the input data and retrieve the results.

References
---
1)  For information on MIDRC GitHub documentation and best practices, please see https://midrc.atlassian.net/wiki/spaces/COMMITTEES/pages/672497665/MIDRC+GitHub+Best+Practices
2)	AI-LAB's Docker standards: https://github.com/ACRCode/AILAB_documentation
3)  Group that developed the algorithm: https://www.acrdsi.org/
