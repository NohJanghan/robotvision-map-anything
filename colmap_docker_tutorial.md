# COLMAP Docker GPU Tutorial

이 문서는 conda COLMAP 대신 Docker COLMAP에서 GPU를 쓰는 최소 절차만 정리한다.

공식 Docker 참고: <https://github.com/colmap/colmap/tree/main/docker>

## 핵심

Docker 안에서 COLMAP이 GPU를 쓰려면 호스트에 다음이 필요하다.

- NVIDIA driver
- Docker 19.03 이상
- NVIDIA Container Toolkit
- GPU 지원 COLMAP Docker image

컨테이너 실행 시 반드시 GPU를 넘긴다.

```bash
docker run --rm --gpus all colmap/colmap:latest colmap -h
```

정상이라면 출력에 `with CUDA`가 보인다.

## 1. GPU 전달 확인

먼저 호스트 GPU가 보이는지 확인한다.

```bash
nvidia-smi
docker --version
```

Docker 컨테이너에서도 GPU가 보이는지 확인한다.

```bash
docker run --rm --gpus all colmap/colmap:latest nvidia-smi
```

이 명령이 실패하면 COLMAP 문제가 아니라 Docker GPU 설정 문제이다.
Ubuntu에서는 NVIDIA Container Toolkit을 설치한 뒤 Docker를 재시작한다.

## 2. 이 프로젝트에서 쓰는 방법

이 프로젝트는 `scripts/colmap/colmap_docker.py` 래퍼를 사용한다.
래퍼는 `docker run --gpus all ... colmap`을 대신 실행한다.

확인:

```bash
scripts/colmap/colmap_docker.py -h
```

정상 출력 예:

```text
COLMAP ... with CUDA
```

Task 1 파이프라인도 기본적으로 이 래퍼를 사용한다.

```bash
conda activate rkv-mapanything
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene \
  --use-gpu 1
```

`--use-gpu 1`은 COLMAP 내부 옵션이다. Docker GPU 전달은 래퍼가 처리한다.

## 3. 공식 Dockerfile로 직접 빌드

공식 이미지를 그대로 써도 되지만, 로컬에서 빌드하려면 다음을 실행한다.

```bash
scripts/colmap/build_colmap_docker.sh
```

이 스크립트는 공식 COLMAP repository를 `third_party/colmap/`에 받고
공식 `docker/Dockerfile`로 `colmap:latest`를 빌드한다.

GPU architecture를 직접 지정하려면 build arg를 넘긴다.

```bash
scripts/colmap/build_colmap_docker.sh --build-arg CUDA_ARCHITECTURES=89
```

빌드 후에는 래퍼가 로컬 `colmap:latest`를 우선 사용한다.

## 4. 유용한 환경변수

```bash
# 사용할 image 직접 지정
COLMAP_DOCKER_IMAGE=colmap/colmap:latest scripts/colmap/colmap_docker.py -h

# 구형 Docker/NVIDIA runtime 방식이 필요한 경우
COLMAP_DOCKER_GPU=runtime scripts/colmap/colmap_docker.py -h

# GPU 전달 없이 CPU 확인만 할 경우
COLMAP_DOCKER_GPU=none scripts/colmap/colmap_docker.py -h
```

보고서나 실험 기록에는 `COLMAP ... with CUDA`가 확인된 실행만 GPU 사용으로 적는다.

## 5. 자주 나는 오류

`could not select device driver`:

- NVIDIA Container Toolkit이 없거나 Docker GPU runtime 설정이 안 된 상태이다.
- `docker run --rm --gpus all colmap/colmap:latest nvidia-smi`부터 통과시킨다.

`COLMAP ... without CUDA`:

- GPU 지원이 없는 COLMAP binary를 실행한 것이다.
- `scripts/colmap/colmap_docker.py -h`로 Docker COLMAP을 쓰는지 확인한다.

Headless OpenGL 오류:

- 기본값은 `QT_QPA_PLATFORM=offscreen`이다.
- 그래도 실패하면 파이프라인에 `--use-xvfb`를 붙인다.

```bash
python scripts/colmap/run_pipeline.py \
  --image-dir data/raw/rgb_sequences/scene \
  --run-name scene \
  --use-gpu 1 \
  --use-xvfb
```
