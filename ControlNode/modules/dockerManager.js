var Docker = require('dockerode');

exports.startAnalyzer = startAnalyzer;

// TODO use args
async function startAnalyzer(host, port) {
    var docker = new Docker();
    var containerSettings = {
        Image: 'docker.io/tensorflow/serving:2.1.4-gpu',
        Cmd: ['--enable_batching'],
        Env: ['MODEL_NAME=mobilenet_v2', 'CUDA_VISIBLE_DEVICES=1'],
        HostConfig: {
                Binds: ["/media/data_1/snva_1/Models/fhwa/tf2/shrp2_nds/active_learning_round_1/mobilenet_v2:/models/mobilenet_v2"],
                PortBindings: {
                    "8500/tcp": [
                        {
                        "HostPort": "8500"
                        }
                    ],
                    "8500/udp": [
                        {
                        "HostPort": "8500"
                        }
                    ],
                    "8501/tcp": [
                        {
                        "HostPort": "8501"
                        }
                    ],
                    "8501/udp": [
                        {
                        "HostPort": "8501"
                        }
                    ]
                },
                Runtime: "nvidia"
            }
    };
    var container = await docker.createContainer(containerSettings);
    await container.start();
    //var container = await docker.run('docker.io/tensorflow/serving:2.1.4-gpu', ['--enable_batching'], process.stdout, containerSettings);
    return container;
}