using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Collections.Specialized.BitVector32;

namespace yolov8test6
{
    public partial class Form1 : Form
    {
        // 클래스 이름 (순서 = 학습된 class id)
        //string[] classNames = { "OK", "TWISTED", "NO_SOLDER", "NO_CHIP", "BRIDGE", "MEOJI", "CHIP_AWAY" };
        string[] classNames = { "OK", "NG" };

        // 클래스별 색상
        //Scalar[] classColors = { Scalar.Lime, Scalar.Red, Scalar.Orange, Scalar.Yellow, Scalar.Blue, Scalar.Navy, Scalar.Purple };
        Scalar[] classColors = { Scalar.Green, Scalar.Red};

        // 추론 쎄션
        static InferenceSession session;
        private const int INPUT_SIZE = 960;

        public Form1()
        {
            InitializeComponent();
            session = new InferenceSession("frt_wire251224.onnx"); //모델 로드
            richTextBox1.AppendText("YOLOv8 모델 로드 완료\n");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image Files|*.jpg;*.png;*.bmp";

            if (ofd.ShowDialog() != DialogResult.OK)
                return;

            // ------------------------------------------
            // OpenCV로 이미지 로드 (Bitmap 사용 ❌)
            // ------------------------------------------
            Mat src = Cv2.ImRead(ofd.FileName);

            // ------------------------------------------
            // YOLO 실행
            // ------------------------------------------
            List<Detection> detections = RunYolo(src);

            // ------------------------------------------
            // 결과 그리기
            // ------------------------------------------
            // RichTextBox에 결과 출력
            // -----------------------------
            richTextBox1.Clear();

            foreach (var det in detections)
            {
                // confidence → 퍼센트 변환
                float percent = det.Score * 100.0f;

                // 좌상단 좌표
                float startX = det.Box.X;
                float startY = det.Box.Y;

                // 중심 좌표 (필요할 때 계산)
                float centerX = det.Box.X + det.Box.Width / 2;
                float centerY = det.Box.Y + det.Box.Height / 2;

                richTextBox1.AppendText(
                    $"Class ID: {det.ClassId} " +
                    $"Score: {percent:0.0}% | " +
                    $"Start(X,Y): ({startX:0}, {startY:0}) | " +
                    $"Center(X,Y): ({centerX:0}, {centerY:0})\n"
                );
            }

            // 결과 그리기

            // ------------------------------------------

            DrawResult(src, detections);

            // ------------------------------------------

            // ------------------------------------------
            // PictureBox에 표시
            // (WinForms는 Bitmap만 표시 가능)
            // ------------------------------------------
            pictureBox1.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);
        }

        private List<Detection> RunYolo(Mat src)
        {
            int orgW = src.Width;
            int orgH = src.Height;

            // 1. Resize 및 RGB 변환
            Mat resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(INPUT_SIZE, INPUT_SIZE));
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            // 2. CHW 형태로 데이터 정규화 및 삽입 (핵심 수정 부분)
            float[] inputData = new float[3 * INPUT_SIZE * INPUT_SIZE];
            int channelStride = INPUT_SIZE * INPUT_SIZE;

            for (int y = 0; y < INPUT_SIZE; y++)
            {
                for (int x = 0; x < INPUT_SIZE; x++)
                {
                    Vec3b pixel = resized.At<Vec3b>(y, x);
                    // Indexing: [Channel * Stride + Y * Width + X]
                    inputData[y * INPUT_SIZE + x] = pixel.Item0 / 255.0f;                   // R
                    inputData[channelStride + y * INPUT_SIZE + x] = pixel.Item1 / 255.0f;   // G
                    inputData[channelStride * 2 + y * INPUT_SIZE + x] = pixel.Item2 / 255.0f; // B
                }
            }

            DenseTensor<float> inputTensor = new DenseTensor<float>(inputData, new[] { 1, 3, INPUT_SIZE, INPUT_SIZE });

            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("images", inputTensor)
    };

            using (var results = session.Run(inputs))
            {
                Tensor<float> output = results.First().AsTensor<float>();
                return ParseOutput(output, orgW, orgH);
            }
        }

        /**
        private List<Detection> RunYolo(Mat src)
        {
            // --------------------------------------------------
            // 1. 원본 이미지 크기 저장
            // (추후 Bounding Box를 원본 좌표로 복원할 때 사용)
            // --------------------------------------------------
            int orgW = src.Width;
            int orgH = src.Height;

            // --------------------------------------------------
            // 2. YOLO 입력 크기로 Resize
            // 보통 YOLOv8 ONNX는 640 x 640 입력
            // --------------------------------------------------
            Mat resized = new Mat();
            Cv2.Resize(src, resized, new OpenCvSharp.Size(INPUT_SIZE, INPUT_SIZE));

            // --------------------------------------------------
            // 3. BGR → RGB 변환
            // OpenCV는 기본이 BGR
            // YOLOv8은 RGB 입력을 사용
            // --------------------------------------------------
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            // --------------------------------------------------
            // 4. Mat → float[] Tensor 데이터 생성
            // Shape: [1, 3, 640, 640]
            // 정규화: 0~255 → 0.0~1.0
            // --------------------------------------------------
            float[] inputData = new float[1 * 3 * INPUT_SIZE * INPUT_SIZE];
            int index = 0;

            for (int y = 0; y < INPUT_SIZE; y++)
            {
                for (int x = 0; x < INPUT_SIZE; x++)
                {
                    // RGB 픽셀 읽기
                    Vec3b pixel = resized.At<Vec3b>(y, x);

                    // YOLO는 CHW 순서
                    //inputData[index++] = pixel.Item0 / 255.0f; // R
                    //inputData[index++] = pixel.Item1 / 255.0f; // G
                    //inputData[index++] = pixel.Item2 / 255.0f; // B
                }
            }

            DenseTensor<float> inputTensor =
                new DenseTensor<float>(inputData,
                    new[] { 1, 3, INPUT_SIZE, INPUT_SIZE });

            // --------------------------------------------------
            // 5. ONNX Runtime 입력 생성
            // 입력 이름은 대부분 "images"
            // --------------------------------------------------
            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            // --------------------------------------------------
            // 6. ONNX Runtime 추론 실행
            // (C# 7.3 → using 블록 필수)
            // --------------------------------------------------
            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                   session.Run(inputs))
            {
                // --------------------------------------------------
                // 7. 출력 Tensor 획득
                // YOLOv8 출력 Shape 예:
                // [1, 8400, 4 + classCount]
                // --------------------------------------------------
                Tensor<float> output = results.First().AsTensor<float>();

                // --------------------------------------------------
                // 8. 결과 파싱 + NMS 적용
                // --------------------------------------------------
                return ParseOutput(output, orgW, orgH);
            }
        }
        **/
        private List<Detection> ParseOutput(Tensor<float> output, int orgW, int orgH)
        {
            List<Detection> detections = new List<Detection>();

            // 디버깅 스크린샷에서 확인된 Dimensions: [1, 11, 18900]
            int numElements = output.Dimensions[1]; // 11 (좌표 4 + 클래스 7)
            int numAnchors = output.Dimensions[2];  // 18900
            int classCount = numElements - 4;       // 7

            for (int i = 0; i < numAnchors; i++)
            {
                float maxScore = 0f;
                int classId = -1;

                // 1. 클래스 점수 추출 (output[0, 4 + c, i] 순서로 접근)
                for (int c = 0; c < classCount; c++)
                {
                    float score = output[0, 4 + c, i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = c;
                    }
                }

                // 임계값 설정 (정확도를 위해 0.4~0.5 권장)
                if (maxScore < 0.4f) continue;

                // 2. 바운딩 박스 좌표 추출 (output[0, 0~3, i] 순서)
                float cx = output[0, 0, i]; // 중심 X
                float cy = output[0, 1, i]; // 중심 Y
                float w = output[0, 2, i];  // 너비
                float h = output[0, 3, i];  // 높이

                // 3. 원본 이미지 크기로 좌표 복원
                float x = (cx - w / 2) * orgW / INPUT_SIZE;
                float y = (cy - h / 2) * orgH / INPUT_SIZE;
                float width = w * orgW / INPUT_SIZE;
                float height = h * orgH / INPUT_SIZE;

                detections.Add(new Detection
                {
                    Box = new RectangleF(x, y, width, height),
                    Score = maxScore,
                    ClassId = classId
                });
            }

            // 중복 제거 후 결과 반환
            return ApplyNms(detections, 0.25f);
        }


        private List<Detection> ApplyNms(List<Detection> detections, float iouThreshold)
        {
            List<Detection> result = new List<Detection>();

            // [수정] 클래스 구분 없이 모든 검출 데이터를 Score(점수) 순으로 정렬
            var sortedDetections = detections
                .OrderByDescending(d => d.Score)
                .ToList();

            while (sortedDetections.Count > 0)
            {
                // 1. 점수가 가장 높은 박스 선택
                var best = sortedDetections[0];
                result.Add(best);
                sortedDetections.RemoveAt(0);

                // 2. 남은 박스들 중 현재 'best' 박스와 많이 겹치는(IoU가 높은) 것들은 
                // 클래스가 다르더라도 모두 제거
                sortedDetections.RemoveAll(d =>
                    CalculateIoU(best.Box, d.Box) > iouThreshold);
            }

            return result;
        }

        private float CalculateIoU(RectangleF a, RectangleF b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);

            float intersectionWidth = Math.Max(0, x2 - x1);
            float intersectionHeight = Math.Max(0, y2 - y1);
            float intersectionArea = intersectionWidth * intersectionHeight;

            float unionArea = a.Width * a.Height + b.Width * b.Height - intersectionArea;

            if (unionArea <= 0)
                return 0;

            return intersectionArea / unionArea;
        }

        private void DrawResult(Mat img, List<Detection> detections)
        {
            // 이전 로그 삭제
            richTextBox1.Clear();

            foreach (var det in detections)
            {
                if (det.ClassId < 0 || det.ClassId >= classNames.Length) continue;

                Scalar color = classColors[det.ClassId];
                string label = classNames[det.ClassId];

                // 1. 이미지에 사각형 그리기 (반드시 int형으로 변환)
                OpenCvSharp.Rect rect = new OpenCvSharp.Rect(
                    (int)det.Box.X, (int)det.Box.Y, (int)det.Box.Width, (int)det.Box.Height);

                // 1. 박스 굵기 수정
                // 마지막 숫자 '2'를 '4'나 '5'로 올리면 더 굵어집니다.
                Cv2.Rectangle(img, rect, color, thickness: 10);

                // 2. 글자 크기 및 굵기 수정
                // HersheySimplex 뒤의 '0.6'은 글자 크기(Scale)
                // 맨 뒤의 '2'는 글자 굵기(Thickness)입니다.
                string text = $"{label}";
                string text2 = $"{det.Score * 100:0.0}%";
                Cv2.PutText(img, text, new OpenCvSharp.Point(rect.X, rect.Y - 10),
                            HersheyFonts.HersheySimplex,
                            fontScale: 2.0, // 글자 크기를 1.0으로 키움
                            color: color,
                            thickness: 7);  // 글자 굵기
                Cv2.PutText(img, text2, new OpenCvSharp.Point(rect.X, rect.Y + rect.Height + 75),
                            HersheyFonts.HersheySimplex,
                            fontScale: 3.0, // 글자 크기를 1.0으로 키움
                            color: color,
                            thickness: 7);  // 글자 굵기

                // 3. RichTextBox에 데이터 기록
                richTextBox1.AppendText($"[ID:{det.ClassId}] {text} | " +
                    $"Pos:({rect.X}, {rect.Y}) W:{rect.Width} H:{rect.Height}\n");
            }
        }
    }

    public class Detection
    {
        public RectangleF Box;   // 바운딩 박스
        public float Score;      // confidence
        public int ClassId;      // 클래스 ID
    }
}
