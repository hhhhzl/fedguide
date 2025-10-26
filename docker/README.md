## Docker Usage

### Install

#### Support
 - MacOS (Apple Silicon)
 - Linux
 - Windows

#### Build Image
```
docker build -f docker/Dockerfile --platform=linux/amd64 -t fedguide .
```
#### Run Container
```
docker run -it -v $(pwd):/workspace/fedguide -w /workspace/fedguide fedguide
```
