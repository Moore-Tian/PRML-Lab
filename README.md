# PRML-Lab 声线转换模型 So-VITS-SVC 的原理研究、复现以及不同编码器下的比较分析
该项目对SVC(Sing Voice Conversion)和VC(Voice Cloning)技术的演进历程进行了回溯，并选择So-VITS-SVC模型作为研究的主要对象。

之后，通过对So-VITS-SVC模型的底层原理进行深入的了解、学习、研究，我们成功地完成了对该模型的复现。

在此基础上，我们使用虚拟角色甘雨的语音数据集对该模型进行了训练，使其能够将输入的人声音频转换为目标声线，并保留原音频的基本内容和音高信息。

此外，我们通过查询资料，尝试了不同的编码器来对模型进行调整，以比较不同编码器对模型的训练与推断效果的影响。
