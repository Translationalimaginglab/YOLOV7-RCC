#!/usr/bin/bash
for f in `seq 1 10`
do
  cat << EOF > "cocoTumorDetectionPre${f}.yaml"
#!/usr/bin/bash

train: /data/AMPrj/Yazdianp/YOLOCutPreDTumor/Train_split_${f}.txt 
val: /data/AMPrj/Yazdianp/YOLOCutPreDTumor/Test_split_${f}.txt 


# number of classes
nc: 1

# class names
names: [ 'Tumor']



EOF
done
