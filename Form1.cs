using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace yolov8test5
{
    public class Detection
    {
        public RectangleF Box;   // 바운딩 박스
        public float Score;      // confidence
        public int ClassId;      // 클래스 ID
    }

    public partial class Form1 : Form
    {
        // 클래스 이름 (순서 = 학습된 class id)
        string[] classNames = { "0", "1", "2", "3", "4", "5", "6" };

        // 클래스별 색상
        Scalar[] classColors = { Scalar.Lime, Scalar.Red, Scalar.Orange, Scalar.Yellow, Scalar.Blue, Scalar.Navy, Scalar.Purple};

        static InferenceSession session;
        private const int INPUT_SIZE = 960;

        public Form1()
        {
            InitializeComponent();
            session = new InferenceSession("chips251215.onnx");
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

            richTextBox1.AppendText($"이미지 로드: {ofd.FileName}\n");

            // ------------------------------------------
            // YOLO 실행
            // ------------------------------------------
            List<Detection> detections = RunYolo(src);

            // ------------------------------------------
            // 결과 그리기
            // ------------------------------------------
            DrawResult(src, detections);

            // ------------------------------------------
            // PictureBox에 표시
            // (WinForms는 Bitmap만 표시 가능)
            // ------------------------------------------
            pictureBox1.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(src);
        }

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
                    inputData[index++] = pixel.Item0 / 255.0f; // R
                    inputData[index++] = pixel.Item1 / 255.0f; // G
                    inputData[index++] = pixel.Item2 / 255.0f; // B
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

        private List<Detection> ParseOutput(Tensor<float> output, int orgW, int orgH)
        {
            List<Detection> detections = new List<Detection>();

            // YOLOv8 ONNX Shape: [1, 11, 8400] (4개 좌표 + 7개 클래스)
            // Dimensions[1] = 11 (변수 개수), Dimensions[2] = 8400 (박스 개수)
            int dimensions = output.Dimensions[1]; // 11
            int numAnchors = output.Dimensions[2]; // 8400 (imgsz 960일 경우 더 많을 수 있음)
            int classCount = dimensions - 4;       // 7

            for (int i = 0; i < numAnchors; i++)
            {
                // -------------------------------------------------------
                // 1. 클래스 점수 추출 및 최대값 찾기
                // -------------------------------------------------------
                float maxScore = 0f;
                int classId = -1;

                for (int c = 0; c < classCount; c++)
                {
                    // 인덱싱 주의: [0, 4 + c, i] 순서로 접근해야 함 (Transpose 대응)
                    float score = output[0, 4 + c, i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = c;
                    }
                }

                if (maxScore < 0.5f) continue;

                // -------------------------------------------------------
                // 2. 좌표 추출 (이것도 [0, 0~3, i] 순서)
                // -------------------------------------------------------
                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                // 원본 좌표 복원 계산
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

            return ApplyNms(detections, 0.45f);
        }

        private List<Detection> ApplyNms(List<Detection> detections, float iouThreshold)
        {
            List<Detection> result = new List<Detection>();

            // -----------------------------
            // 클래스별로 NMS 수행
            // -----------------------------
            foreach (int classId in detections.Select(d => d.ClassId).Distinct())
            {
                var classDetections = detections
                    .Where(d => d.ClassId == classId)
                    .OrderByDescending(d => d.Score)
                    .ToList();

                while (classDetections.Count > 0)
                {
                    // 가장 score 높은 박스 선택
                    var best = classDetections[0];
                    result.Add(best);
                    classDetections.RemoveAt(0);

                    // IoU가 threshold 이상인 박스 제거
                    classDetections.RemoveAll(d =>
                        CalculateIoU(best.Box, d.Box) > iouThreshold);
                }
            }

            richTextBox1.AppendText($"NMS 후 검출 수: {result.Count}\n");
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
            // 클래스별 색상 배열 (Scalar)
            foreach (var det in detections)
            {
                // 클래스 범위 체크
                if (det.ClassId < 0 || det.ClassId >= classNames.Length)
                {
                    richTextBox1.AppendText(
                        $"[경고] classId {det.ClassId} 는 정의되지 않음\n");
                    continue;
                }

                // 클래스별 색상 / 이름
                Scalar color = classColors[det.ClassId];
                string label = classNames[det.ClassId];

                // -----------------------------
                // 바운딩 박스
                // -----------------------------
                Cv2.Rectangle(
                    img,
                    new OpenCvSharp.Rect(
                        (int)det.Box.X,
                        (int)det.Box.Y,
                        (int)det.Box.Width,
                        (int)det.Box.Height),
                    color,
                    2);

                // -----------------------------
                // 텍스트 (클래스명 + score)
                // -----------------------------
                Cv2.PutText(
                    img,
                    $"{label} ({det.Score:0.00})",
                    new OpenCvSharp.Point(
                        det.Box.X,
                        det.Box.Y - 5),
                    HersheyFonts.HersheySimplex,
                    0.6,
                    color,
                    2);

                // RichTextBox 출력
                richTextBox1.AppendText(
                    $"[{label}] Score:{det.Score:0.00} " +
                    $"X:{det.Box.X:0} Y:{det.Box.Y:0}\n");
            }
        }

    }

}
