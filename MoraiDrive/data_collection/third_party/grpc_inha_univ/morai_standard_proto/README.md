## Morai Standard Proto
### - Summary
- Morai의 각종 툴 체인에서 사용되고 있는 gRPC를 위한 통합 Proto를 정의 하는 공간 입니다.
- 다음과 같은 곳에서 사용되고 있습니다. 
    - Morai Sim V1
    - Scenario Runner
    - DataGen
    - 외부 고객사
        - Naver
        - Ansys        


### - References
- [gRPC official site](https://grpc.io/)  
- [Protocol Buffers Documentation site](https://protobuf.dev/)  
- [각종 Interface Team 메모](https://morai.atlassian.net/wiki/spaces/MTG/pages/edit-v2/1327818157)  

### - How to Use
1. proto 정의 (proto 자체는 언어와 상관 없이 공통으로 사용할 수 있음)
2. 사용할 개발 언어 선택 (python, csharp, c++ ....)
3. 개발환경 구축    
4. proto -> Code Generation = Build proto
5. 생성된 파일을 개발 프로젝트에 첨부하여 사용    

### - How to Builds

#### [Python]
- Requirements
- 2023.10.19일 기준
```
- python 3.7 or higher
- grpcio 1.44.0 
- grpcio-tools 1.44.0
- protobuf 4.24.4
```
[설치 방법은 공식 사이트 참고](https://grpc.io/docs/languages/python/quickstart/)

```
git clone http://172.16.1.20:8081/morai-interface/morai_standard_proto
cd morai_standard_proto
python build_proto.py
```

#### [.Net]
- Requirements
```
Nuget 일단 확인된 정보
- Google.Protobuf 3.21.9
- Grpc.Core 2.46.5
- Grpc.Core.Api 2.46.5
- Grpc.tools 2.38.1
```

- [visual studio 를 이용하는 방법](https://morai.atlassian.net/wiki/spaces/MTG/pages/1525778402/IF+Visual+Studio+gRPC)
    - 추가 정리 필요
- 배치 파일로 하는 방법
```
git clone http://172.16.1.20:8081/morai-interface/morai_standard_proto
cd morai_standard_proto
.\make_env.sh
```
