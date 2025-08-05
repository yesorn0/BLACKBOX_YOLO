# 🧃 YOLO모델을 활용한 음성알림 BLACK BOX

## 시연
| | |
|---|---|
| <img width="400" alt="image" src="https://github.com/user-attachments/assets/3a034e86-80fa-4720-b4c1-deb45dd42745" /> | <img width="400" alt="image" src="https://github.com/user-attachments/assets/8bd9fdc3-43ea-41ca-98db-e13dba98ba14" /> |




## 개요
신호 정체시 휴대폰 사용 등 여러 이유로 신호변경을 인지하지 못한 운전자로 인해 교차로 정체 발생
청각으로 신호 변경을 알리는 시스템 필요

- 개발기간 : 25. 06. 16 ~ 25. 06 . 22
- 팀장 : 오준택
- 팀원 : 이종희, 안진홍
- 담당역할
  - 모델 제작(YOLO기반 전이학습)
  - TTS 제작

## 프로젝트 핵심 기능
- 실시간 도로시설물 탐지 : 도로시설물 인식
- 음성 경고 시스템 : TTS 기반 실시간 위험 알림
- 블랙박스 기능 : 영상저장 및 객체 탐지 결과 기록
- 실시간 처리 : 웹캠/비디오 실시간 추론

## 시스템 구성
<img width="400" alt="image" src="https://github.com/user-attachments/assets/f90d5dfd-f387-4035-a570-b426adf77d9a" />


## 사용기법
### 딥러닝 모델링
YOLO 전이 학습
- YOLOv8m 모델 기반 커스텀 학습
데이터 전처리
- COCO -> YOLO변환 : JSON어노테이션을 YOLO.txt 형식으로 변환
- 폴리곤 ->BBox 변환 : 세그멘테이션 좌표를 바운딩 박스로 변환

데이터 증강
- 기하학적 변환 : 플립, 회전, 스케일링, 이동
- 색상변환 : HSV 색상 공간 조정
- 모자이크 증강 : 4개 이미지 결합으로 다양성 증대
- Mixup & CopyPaste 

실시간 웹캠 처리
- Opencv 연동 : 웹캠 스트림 실시간 처리

## 데이터셋 정보
### AI Hub - 전국 도로시설물 영상정보 데이터
- 클래스 : 34개 도로시설물 카테고리
- 교통안전시설 : 시선유도표지, 갈매기 표지 등
- 구조물 : 교량, 터널, 지하차도, 고가차도 등
- 보안시설 : 도로전광표시, 긴급연락시설 등
