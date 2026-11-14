import 'package:camera/camera.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:material_ui/material_ui.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await _setupModelInitially();
  final cameras = await availableCameras();
  final backCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.back,
  );

  runApp(MyApp(camera: backCamera));
}

Future<void> _setupModelInitially() async {
  await FlutterEdgeAi.initialize(
    inferenceEngines: [LiteRtLmEngine()],
    huggingFaceToken:
        const String.fromEnvironment("HUGGINGFACE_TOKEN").isNotEmpty
        ? const String.fromEnvironment("HUGGINGFACE_TOKEN")
        : null,
    maxDownloadRetries: 10,
  );

  final resolved = await FlutterEdgeAi.resolveHuggingFace(
    'mlboydaisuke/OvisOCR2-LiteRT',
    fileType: ModelFileType.litertlm,
  );

  await FlutterEdgeAi.installModel(
    modelType: resolved.modelType ?? ModelType.general,
    fileType: ModelFileType.litertlm,
  ).fromNetwork(resolved.url).withProgress((progress) {
    // TODO: show something more visual
    print('Download: $progress%.');
  }).install();
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: CameraScreen(camera: camera));
  }
}

class CameraScreen extends StatefulWidget {
  final CameraDescription camera;

  const CameraScreen({super.key, required this.camera});

  @override
  State<StatefulWidget> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController controller;
  late Future<void> initControllerFuture;

  @override
  void initState() {
    super.initState();

    controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
    );

    initControllerFuture = controller.initialize();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder<void>(
        future: initControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == .done) {
            return Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),

                Positioned(
                  bottom: 40,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: FloatingActionButton(
                      onPressed: () => showResultPopup(context),
                      child: const Icon(Icons.camera_alt),
                    ),
                  ),
                ),
              ],
            );
          }

          if (snapshot.hasError) {
            return Center(child: Text("Camera error: ${snapshot.error}"));
          }

          return const Center(child: CircularProgressIndicator());
        },
      ),
    );
  }

  Future<String> takePicture() async {
    InferenceModelSession? session;

    try {
      await initControllerFuture;

      final image = await controller.takePicture();
      final imageBytes = await image.readAsBytes();

      // TODO: cache this
      final model = await FlutterEdgeAi.getActiveModel(
        maxTokens: 2048,
        preferredBackend: PreferredBackend.gpu,
      );

      session = await model.openSession(
        temperature: 0,
        maxOutputTokens: 2048,
        enableVisionModality: true,
        enableThinking: false,
      );

      await session.addQueryChunk(
        Message.withImage(
          text: """
Extract all information from the main body of the
document image and represent it in plain text format.

Ignore headers and footers.
Preserve the original reading order.
Do not summarize or omit information.
""",
          imageBytes: imageBytes,
          isUser: true,
        ),
      );

      final text = await session.getResponse();
      return text;
    } catch (e) {
      debugPrint("Couldn't take a picture! $e");
      rethrow;
    } finally {
      await session?.close();
    }
  }

  void showResultPopup(BuildContext context) {
    final dragController = DraggableScrollableController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.12,
        minChildSize: 0.12,
        maxChildSize: 0.85,
        snap: true,
        snapSizes: const [0.12, 0.85],
        builder: (context, scrollController) => FutureBuilder<String>(
          future: takePicture(),
          builder: (context, snapshot) {
            if (snapshot.connectionState == .done) {
              // Automatically expand when completed
              WidgetsBinding.instance.addPostFrameCallback((_) {
                if (dragController.isAttached) {
                  dragController.animateTo(
                    0.85,
                    duration: const Duration(milliseconds: 500),
                    curve: Curves.easeOut,
                  );
                }
              });
            }

            return Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: .vertical(top: .circular(24)),
              ),
              child: ListView(
                controller: scrollController,
                padding: .all(20),
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 5,
                      decoration: BoxDecoration(
                        color: Colors.grey.shade400,
                        borderRadius: .circular(10),
                      ),
                    ),
                  ),
                  SizedBox(height: 25),

                  if (snapshot.connectionState == .waiting) ...[
                    Row(
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(width: 15),
                        Text("Scanning..."),
                      ],
                    ),
                  ] else if (snapshot.hasError) ...[
                    Text("Error: ${snapshot.error}"),
                  ] else ...[
                    Text(
                      snapshot.data ?? "No text detected",
                      style: const TextStyle(fontSize: 18),
                    ),
                  ],
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
