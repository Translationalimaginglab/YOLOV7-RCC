import os
import SimpleITK as sitk
import numpy as np
import json
import functools
import operator
from RandomSplit import RandomSplit
from rcc_common import LoadMask

np.random.seed(727)

def _connected_components(npMask):
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    ccMask = ccFilter.Execute(sitk.GetImageFromArray(npMask))
    objCount = ccFilter.GetObjectCount()

    return sitk.GetArrayFromImage(ccMask), objCount

def _remove_small_connected_components(npMask, small=15):
    npCcMask, objectCount = _connected_components(npMask)

    for label in range(1, objectCount+1):
        if (npCcMask == label).sum() < small:
            npMask[npCcMask == label] = 0

    return npMask

def _count_tumors(npMask):
    npTumorMask = (npMask == 3).astype(np.uint8)
    npTumorMask = _remove_small_connected_components(npTumorMask)

    if not npTumorMask.any():
        return 0

    _, objCount = _connected_components(npTumorMask)

    return objCount

dataRoot="/data/AMPrj/NiftiCombinedNew"
p = 0.8
with open('allPath.txt', mode="rt", newline="") as f:
    data = [ case.strip() for case in f if len(case.strip()) > 0 ]


tumorTypes = dict()
caseMap = dict()
invCaseMap = dict()
tumorMap = dict()

for case in data:
    caseId = os.path.join(case.split('/')[0],case.split('/')[1])
    patientId = case.split('/')[0]

    if patientId in caseMap:
        i = caseMap[patientId]
    else:
        i = len(caseMap)
        caseMap[patientId] = i

    invCaseMap.setdefault(i, []).append(caseId)
    #invCaseMap.append(patientId)

    tumorType=case.split('/')[2]
    tumorTypes.setdefault(tumorType, []).append(caseId)

#tumorTypeMap = { tumorType: i for i, tumorType in enumerate(tumorTypes) }

for key, cases in tumorTypes.items():
    print(f"{key}: {len(cases)}")

# Pathologies for left/right kidney, tumor burden for left/right and self count guide
K = 2*len(tumorTypes) + 2*2 + 1
N = len(caseMap)

W = np.zeros([K, N])

rightKidneyVolumeIndex = 2*len(tumorTypes)
rightTumorVolumeIndex = rightKidneyVolumeIndex+1
leftKidneyVolumeIndex = rightTumorVolumeIndex+1
leftTumorVolumeIndex = leftKidneyVolumeIndex+1
selfIndex = leftTumorVolumeIndex+1

# Each instance itself counts as 1 instance
#W[selfIndex, :] = 1
for i in range(N):
    W[selfIndex, i] = len(invCaseMap[i])

caseCount = 0

for tumorIndex, tumorType in enumerate(tumorTypes):
    cases = tumorTypes[tumorType]

    rightTumorIndex = tumorIndex
    leftTumorIndex = tumorIndex + len(tumorTypes)
    
    for case in cases:
        print(case)
        caseCount += 1

        patientId = case.split('/')[0]

        caseIndex = caseMap[patientId]

        caseDir = os.path.join(dataRoot, 'Masks', case)
        maskPath = os.path.join(caseDir, "mask_aligned.nii.gz")
        #print (caseDir)
        if not os.path.exists(caseDir):
            print (f'path not exist: {caseDir}')
            continue
        if not os.path.exists(maskPath):
            print (f'Mask not exist: {caseDir}')
            continue
        #mask = sitk.ReadImage(maskPath)
        mask = LoadMask(maskPath)
        voxelVolume = functools.reduce(operator.mul, mask.GetSpacing())
        npMask = sitk.GetArrayFromImage(mask)
        halfX = npMask.shape[-1]//2

        #print(np.unique(npMask))

        npRightMask, npLeftMask = npMask[..., :halfX], npMask[..., halfX:]

        rightKidneyVolume = voxelVolume*(npRightMask == 1).sum()
        leftKidneyVolume = voxelVolume*(npLeftMask == 1).sum()
        rightTumorVolume = voxelVolume*(npRightMask == 3).sum()
        leftTumorVolume = voxelVolume*(npLeftMask == 3).sum()

        rightTumorCount = _count_tumors(npRightMask)
        leftTumorCount = _count_tumors(npLeftMask)

        #print(rightTumorCount)
        #print(leftTumorCount)
        #print(rightTumorVolume)
        #print(leftTumorVolume)

        W[rightKidneyVolumeIndex, caseIndex] += rightKidneyVolume
        W[rightTumorVolumeIndex, caseIndex] += rightTumorVolume

        W[leftKidneyVolumeIndex, caseIndex] += leftKidneyVolume
        W[leftTumorVolumeIndex, caseIndex] += leftTumorVolume

        W[rightTumorIndex, caseIndex] += rightTumorCount
        W[leftTumorIndex, caseIndex] += leftTumorCount

def PrettyPrint(v, title=None):
    if title is not None:
        print(f"\n *** {title} ***\n")

    for tumorIndex, tumorType in enumerate(tumorTypes):
        rightTumorIndex = tumorIndex
        leftTumorIndex = tumorIndex + len(tumorTypes)

        print(f"right tumor type '{tumorType}' total: {v[rightTumorIndex]}")
        print(f"left tumor type '{tumorType}' total: {v[leftTumorIndex]}")

    print(f"right kidney volume total: {v[rightKidneyVolumeIndex]}")
    print(f"left kidney volume total: {v[leftKidneyVolumeIndex]}")
    print(f"right tumor volume total: {v[rightTumorVolumeIndex]}")
    print(f"left tumor volume total: {v[leftTumorVolumeIndex]}")
    print(f"instance count total: {v[selfIndex]}")

np.savetxt("rcc_W.csv", W, delimiter=',')
for i in range (10):
    xtrain, res = RandomSplit(W, p, tries=10000)

    PrettyPrint(W.sum(axis=1), "Total")
    #print((p*W.sum(axis=1)))
    PrettyPrint((p*W.sum(axis=1)), "Expected")
    #print(np.inner(W, xtrain))
    PrettyPrint(np.inner(W, xtrain), "SVD Split")
    print(f"res = {res}")

    with open(f"new_training80{i}.txt", mode="wt", newline="") as f, open(f"new_testing20{i}.txt", mode="wt", newline="") as g:
        for i in range(N):
            cases = invCaseMap[i]

            if xtrain[i]:
                for case in cases:
                    f.write(f"{case}\n") 
            else:
                for case in cases:
                    g.write(f"{case}\n")

print("Done.")

