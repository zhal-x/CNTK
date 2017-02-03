test_cases = {
    "ResNet110::py_2-worker": {
        "dir": "Examples/Image/Classification/ResNet/Python",
        "exe": "python",
        "args": ["TrainResNet_CIFAR10.py", "-n", "resnet110"],
        "distributed": ["-n", "2"]
    },
    "ResNet110::bs_2-worker": {
        "dir": "Examples/Image/Classification/ResNet/BrainScript",
        "exe": "cntk",
        "args": ["configFile=ResNet110_CIFAR10.cntk"],
        "distributed": ["-n", "2"]
    },
}