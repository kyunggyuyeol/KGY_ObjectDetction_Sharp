using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using TensorFlow;

using System.Net;
using ICSharpCode.SharpZipLib.Tar;
using ICSharpCode.SharpZipLib.GZip;
using System.Diagnostics;

namespace objectdectTest
{
    public partial class Form1 : Form
    {

        private static IEnumerable<CatalogItem> _catalog;
        private string _currentDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        private string _input = "";
        private string _output = "";
        private string _catalogPath = "";
        private string _modelPath = "";


        //Path.Combine(_currentDir, "test_images/outputimage.jpg");

        private static double MIN_SCORE_FOR_OBJECT_HIGHLIGHTING = 0.5;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();

            _input = Path.Combine(_currentDir, "test_images/input.jpg");
            _output = Path.Combine(_currentDir, "test_images/output.jpg");
            _catalogPath = Path.Combine(_currentDir, @"data/mscoco_label_map.pbtxt");
            _modelPath = Path.Combine(_currentDir, @"data/frozen_inference_graph.pb");


            //cat load
            _catalog = CatalogUtil.ReadCatalogItems(_catalogPath);

            var fileTuples = new List<(string input, string output)>() { (_input, _output) };

            //modelpath
            string modelFile = _modelPath;

            using (var graph = new TFGraph())
            {
                var model = File.ReadAllBytes(modelFile);
                graph.Import(new TFBuffer(model));

                using (var session = new TFSession(graph))
                {
                    foreach (var tuple in fileTuples)
                    {
                        var tensor = ImageUtil.CreateTensorFromImageFile(tuple.input, TFDataType.UInt8);
                        var runner = session.GetRunner();

                        runner
                            .AddInput(graph["image_tensor"][0], tensor)
                            .Fetch(
                            graph["detection_boxes"][0],
                            graph["detection_scores"][0],
                            graph["detection_classes"][0],
                            graph["num_detections"][0]);

                        var output = runner.Run();
                        var boxes = (float[,,])output[0].GetValue(jagged: false);
                        var scores = (float[,])output[1].GetValue(jagged: false);
                        var classes = (float[,])output[2].GetValue(jagged: false);
                        var num = (float[])output[3].GetValue(jagged: false);

                        DrawBoxes(boxes, scores, classes, tuple.input, tuple.output, MIN_SCORE_FOR_OBJECT_HIGHLIGHTING);
                    }
                }
            }

            watch.Stop();
            var data = watch.Elapsed;
        }

        private void DrawBoxes(float[,,] boxes, float[,] scores, float[,] classes, string inputFile, string outputFile, double minScore)
        {
            var x = boxes.GetLength(0);
            var y = boxes.GetLength(1);
            var z = boxes.GetLength(2);

            float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

            using (var editor = new ImageEditor(inputFile, outputFile))
            {
                for (int i = 0; i < x; i++)
                {
                    for (int j = 0; j < y; j++)
                    {
                        if (scores[i, j] < minScore) continue;

                        for (int k = 0; k < z; k++)
                        {
                            var box = boxes[i, j, k];
                            switch (k)
                            {
                                case 0:
                                    ymin = box;
                                    break;
                                case 1:
                                    xmin = box;
                                    break;
                                case 2:
                                    ymax = box;
                                    break;
                                case 3:
                                    xmax = box;
                                    break;
                            }

                        }

                        int value = Convert.ToInt32(classes[i, j]);
                        CatalogItem catalogItem = _catalog.FirstOrDefault(item => item.Id == value);
                        editor.AddBox(xmin, xmax, ymin, ymax, $"{catalogItem.DisplayName} : {(scores[i, j] * 100).ToString("0")}%");
                    }
                }
            }
        }


    }
}
