# Image Detection in Noisy Images

## Abstract 
The problem of image noise is prevalent in various visual recognition tasks. This problem arises due to multiple factors like insufficient illumination, low-quality camera sensors, background-clutters, etc. Low-quality images incur degraded performance for visual recognition tasks like classification, object detection, because of the noise present in the image. Thus, mostly the latest image recognition benchmarks and methods, such as Pascal Visual Object Classes Challenges and Microsoft Common Objects in Context Challenges, rely primarily on clean and high-definition images. Furthermore, image noise is a typical problem in real-time applications like surveillance, autonomous vehicles, etc. To tackle this bothersome problem of underperformance of visual recognition tasks, particularly object detection, due to inclusion of noise in real-time scenarios, a technique has been proposed to reduce the image noise's impact on the object detection task and improve its performance altogether. A two-phase pipeline is introduced, in which the first phase consists ofremoving noise from the low-quality image, and the second stage includes accomplishing the object detection process by employing the standard Single-shot multibox object detector. The proposed approach is evaluated on the Pascal Visual Object Classes benchmark. The detection results and image denoising results exhibit the efficiency of the proposed technique for reliable object detection with varying image noise levels.

## Citation
For more details, please read our complete paper [Image Detection in Noisy Images](https://ieeexplore.ieee.org/document/9432243).

If you use any part of our findings, please cite us using:
```bibtex
@INPROCEEDINGS{9432243,
  author={Yadav, Kushagra and Mohan, Dakshit and Parihar, Anil Singh},
  booktitle={2021 5th International Conference on Intelligent Computing and Control Systems (ICICCS)}, 
  title={Image Detection in Noisy Images}, 
  year={2021},
  volume={},
  number={},
  pages={917-923},
  keywords={Visualization;Image recognition;Surveillance;Object detection;Benchmark testing;Real-time systems;Sensors;Image denoising;Object detection;Deep learning;Residual learning},
  doi={10.1109/ICICCS51141.2021.9432243}}
```
