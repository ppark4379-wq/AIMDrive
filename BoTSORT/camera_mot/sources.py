"""Transport selection: UDP by default, ROS only when explicitly requested."""


def create_source(config, transport=None):
    transport = transport or config.get('transport', 'udp')
    if transport == 'udp':
        if any('udp_port' not in value for value in config['cameras'].values()):
            raise ValueError('Every UDP camera requires udp_port')
        from camera_mot.udp_source import UdpCameraSource
        return UdpCameraSource(config['cameras'], config.get('bind_ip', '0.0.0.0'),
                               config.get('sender_ip'), config.get('index_base', 0))
    if transport == 'ros':
        if not config.get('topic'):
            raise ValueError('ROS source requires topic')
        from camera_mot.ros_source import RosCameraSource
        return RosCameraSource(config['cameras'], config['topic'])
    raise ValueError('Unsupported camera transport: ' + str(transport))
