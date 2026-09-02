import os, sys, time
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
sys.path.append(os.path.normpath(os.path.join(current_path, 'proto')))
from proto.sim_adapter import *

MORAI_SIM_ADDRESS = '127.0.0.1'
MORAI_SIM_PORT = 7789

"""
MORAI gRPC API 사용예제
자세한 설명 아래 gRPC Manual Link를 참조한다.
https://help-morai-sim.scrollhelp.site/ko/sim-api-guide/Working-version/grpc-api-docs
https://morai-autonomous.github.io/grpc-docs/


MORAI SIM을 실행(로딩 완료)후 본 예제코드를 실행함.    
"""

class MORAI_gRPC:

    def __init__(self):        
        self.adaptor = SimAdapter()
        self.adaptor.connect(MORAI_SIM_ADDRESS, MORAI_SIM_PORT)


    def main(self):                
        self.SetVehiclePosition()
        time.sleep(15)
        self.SetVehicleControlMode()
    
        

    def SetVehiclePosition(self):
        """
        Ego 차량 위치 Setting        
        """
        from proto.morai.common.enum_pb2 import ObjectType        
        from proto.morai.actor.actor_set_pb2 import SetTransformParam
        request = SetTransformParam()
        request.actor_info.id.value = 'Ego'
        request.actor_info.object_type = ObjectType.OBJECT_TYPE_VEHICLE
        request.actor_info.client_key = 'Morai_Example'
        request.transform.location.x = 28.22
        request.transform.location.y = 1117.90
        request.transform.location.z = 0.79
        request.transform.rotation.x = 0.78
        request.transform.rotation.y = 1.351
        request.transform.rotation.z = 3.320

        try:
            response = self.adaptor._actor_stub.SetTransform(request)
            print(f'SetVehiclePosition Response : {response.description}')
        except Exception as e :
            print(f'SetVehiclePosition Error : {e}')
        



    def SetVehicleControlMode(self):
        from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam
        from proto.morai.actor.actor_enum_pb2 import VehicleControlMode
        from proto.morai.common.enum_pb2 import ObjectType
        """
        Ego 차량의 ControlMode 선택.
        request mode 
        VehicleControlMode.VEHICLE_CONTROL_KEYBOARD = 키보드 제어
        VehicleControlMode.VEHICLE_CONTROL_AUTO_MODE = 외부 제어(알고리즘)
        VehicleControlMode.VEHICLE_CONTROL_CRUISE_MODE = MORAI 내부 제어
        """
        
        request = VehicleControlModeParam()
        request.actor_info.id.value = 'Ego'
        request.actor_info.object_type = ObjectType.OBJECT_TYPE_VEHICLE
        request.actor_info.client_key = 'Morai_Example'
        request.mode = VehicleControlMode.VEHICLE_CONTROL_CRUISE_MODE

        try:
            response = self.adaptor._actor_stub.SetVehicleControlMode(request)
            print(f'SetVehicleControlMode Response : {response.description}')
        except Exception as e :
            print(f'SetVehicleControlMode Error : {e}')

        
if __name__ == '__main__':

    example = MORAI_gRPC()    
    example.main()
