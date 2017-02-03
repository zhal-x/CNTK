test_cases = {
    "MLP::py": {
        "dir": "Examples/Image/Classification/MLP/Python",
        "exe": "python",
        "args": ["SimpleMNIST.py"]
    },
    "MLP::bs": {
        "dir": "Examples/Image/GettingStarted",
        "exe": "cntk",
        "args": ["configFile=01_OneHidden.cntk"]
    },
    "ResNet20::py": {
        "dir": "Examples/Image/Classification/ResNet/Python",
        "exe": "python",
        "args": ["TrainResNet_CIFAR10.py"]
    },
    "ResNet20::bs": {
        "dir": "Examples/Image/Classification/ResNet/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ResNet20_CIFAR10.cntk"]
    },
    "ResNet110::py": {
        "dir": "Examples/Image/Classification/ResNet/Python",
        "exe": "python",
        "args": ["TrainResNet_CIFAR10.py", "-n", "resnet110"]
    },
    "ResNet110::bs": {
        "dir": "Examples/Image/Classification/ResNet/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ResNet110_CIFAR10.cntk"]
    },
    "SLU::py": {
        "dir": "Examples/LanguageUnderstanding/ATIS/Python",
        "exe": "python",
        "args": ["LanguageUnderstanding.py"]
    },
    "SLU::bs": {
        "dir": "Examples/LanguageUnderstanding/ATIS/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ATIS.cntk"]
    },
    "S2S::py": {
        "dir": "Examples/SequenceToSequence/CMUDict/Python",
        "exe": "python",
        "args": ["Sequence2Sequence.py"]
    },
}