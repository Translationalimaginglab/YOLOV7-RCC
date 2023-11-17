#!/usr/bin/bash
for f in `seq 1 10`
do
  cat << EOF > "cocoTumorDetectionCT${f}.yaml"
#!/usr/bin/bash

train: /data/AMPrj/kits19/YOLO/Train_split_${f}.txt 
val: /data/AMPrj/kits19/YOLO/Test_split_${f}.txt 

# number of classes
nc: 1

# class names
names: ['Tumor' ]



EOF
done
