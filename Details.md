# 🧑‍🎓 ASAP 论文解读 
ASAP取自Aligning Simulation and Real Physics
## 💡 Overview
A two-stage framework

1️⃣ stage: 
- pre-trained motion tracking policies

2️⃣ stage: 
- pre-trained policy deployment 
- real-world data  collection 
- train a delta action model

## 💡 Details

### Stage 1️⃣（预训练）：
**Part A**  Data Generation  
**Step 1:** Exploiting __TRAM__ to transform collected human demonstrated videos as __SMPL__ format.

>**Contour:**.   
⚙️**SMPL** parameters contains __[beta(shape), theta(joint_pose), R&T(body_pose)]__.  
**Shape** describes the appearance like size, height, etc.  
**Joint_pose** describes the respective joint angle.  
**R&T** desrcibes a gloal rotation and orientaton of the whole body.

**Step 2:** Data processing to filter nosie and erros. Raw data refers to directly collected data from __TRAM__. Then __MaskedMimic__ employed to imitate raw data produces the output as the cleaned data. 

>❗️❗️**Note:** __MaskedMimic__ can only validate the collected motion but not filter noise. __MaskedMimic__ is jsut a motion tracker so only the motion tracked by tracker can be considered as a valid(cleaned) motion.

**Step 3:** Retarget motion. Aligning from human collected data to humanoid enable train robot. 

**——————————————————————————————————————————————————**
**Part B** Motion Tracking Policy Training


