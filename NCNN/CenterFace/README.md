# CenterFace: Face as Point

## 测试环境
- ncnn
- opencv
- [prj-ncnn](https://github.com/Star-Clouds/CenterFace/tree/master/prj-ncnn)

## 运行 prj-ncnn
```Shell
git clone https://github.com/Star-Clouds/CenterFace/
cd centerface/prj-ncnn
vi CMakeList.txt
```
```
<modify>
Set(DIR “/ncnn/build/install”)
...
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
<modify>
```
```Shell
cmake ./
make
./demo ../models/ncnn sample.jpg
```

## References
- [CenterFace](https://github.com/Star-Clouds/CenterFace)