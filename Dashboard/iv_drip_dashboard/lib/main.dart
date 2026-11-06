import 'dart:convert';
import 'dart:io' show File;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'screens/login_page.dart';

void main() {
  runApp(const IVDripApp());
}

class IVDripApp extends StatelessWidget {
  const IVDripApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IV Drip Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F766E),
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F7FA),
      ),
      home: const LoginPage(),
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  // Flutter Web on this computer uses localhost.
  // Android emulator should use 10.0.2.2 instead.
  static const String apiBaseUrl = 'http://127.0.0.1:8000';

  bool apiOnline = false;
  bool checkingStatus = false;
  bool predicting = false;

  String? predictionLabel;
  double? confidence;
  String? predictionTime;
  String? errorMessage;

  XFile? selectedImage;
  final ImagePicker picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    checkApiStatus();
  }

  Future<void> checkApiStatus() async {
    setState(() => checkingStatus = true);

    try {
      final response = await http
          .get(Uri.parse('$apiBaseUrl/api/status'))
          .timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          apiOnline = data['api_status'] == 'online' &&
              data['model_loaded'] == true;
          errorMessage = null;
        });
      } else {
        setState(() {
          apiOnline = false;
          errorMessage = 'API returned ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        apiOnline = false;
        errorMessage = 'Cannot connect to backend. Is it running?';
      });
    } finally {
      if (mounted) setState(() => checkingStatus = false);
    }
  }

  Future<void> chooseImage() async {
    try {
      final image = await picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        setState(() {
          selectedImage = image;
          errorMessage = null;
          predictionLabel = null;
          confidence = null;
        });
      }
    } catch (e) {
      setState(() => errorMessage = 'Could not select image: $e');
    }
  }

  Future<void> predictImage() async {
    if (selectedImage == null) {
      setState(() => errorMessage = 'Choose a saline-bottle image first.');
      return;
    }

    setState(() {
      predicting = true;
      errorMessage = null;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBaseUrl/api/predict'),
      );

      if (kIsWeb) {
        final bytes = await selectedImage!.readAsBytes();
        request.files.add(
          http.MultipartFile.fromBytes(
            'file',
            bytes,
            filename: selectedImage!.name,
          ),
        );
      } else {
        request.files.add(
          await http.MultipartFile.fromPath('file', selectedImage!.path),
        );
      }

      final streamedResponse =
          await request.send().timeout(const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          predictionLabel = data['label']?.toString();
          confidence = (data['confidence'] as num?)?.toDouble();
          predictionTime = data['timestamp']?.toString();
        });
      } else {
        setState(() {
          errorMessage =
              'Prediction failed (${response.statusCode}): ${response.body}';
        });
      }
    } catch (e) {
      setState(() => errorMessage = 'Prediction request failed: $e');
    } finally {
      if (mounted) setState(() => predicting = false);
    }
  }

  String prettyLabel(String? label) {
    if (label == null) return 'No prediction yet';
    switch (label) {
      case 'sal_data_100':
        return 'Full · 100%';
      case 'sal_data_80':
        return 'Level · 80%';
      case 'sal_data_50':
        return 'Level · 50%';
      case 'sal_data_empty':
        return 'Empty';
      default:
        return label.replaceAll('sal_data_', '');
    }
  }

  @override
  Widget build(BuildContext context) {
    final percent =
        confidence == null ? '--' : '${(confidence! * 100).toStringAsFixed(1)}%';

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'IV Drip Monitor',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            tooltip: 'Refresh backend status',
            onPressed: checkApiStatus,
            icon: const Icon(Icons.refresh),
          ),
          const SizedBox(width: 12),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1100),
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              Row(
                children: [
                  const Expanded(
                    child: Text(
                      'Monitoring Dashboard',
                      style:
                          TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
                    ),
                  ),
                  _StatusChip(
                    online: apiOnline,
                    loading: checkingStatus,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                'Image-based IV fluid level classification',
                style: TextStyle(color: Colors.black54),
              ),
              const SizedBox(height: 24),

              // Backend/model connection card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Row(
                    children: [
                      Icon(
                        apiOnline ? Icons.cloud_done : Icons.cloud_off,
                        size: 36,
                        color: apiOnline ? Colors.teal : Colors.redAccent,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Backend connection',
                              style: TextStyle(
                                  fontWeight: FontWeight.bold, fontSize: 16),
                            ),
                            const SizedBox(height: 4),
                            Text(apiOnline
                                ? 'API online · Model loaded'
                                : errorMessage ?? 'Backend unavailable'),
                          ],
                        ),
                      ),
                      OutlinedButton(
                        onPressed: checkingStatus ? null : checkApiStatus,
                        child: const Text('Check'),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // Image upload and prediction
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Bottle image analysis',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 6),
                      const Text(
                        'Select an image to send it to the trained model.',
                        style: TextStyle(color: Colors.black54),
                      ),
                      const SizedBox(height: 16),
                      if (selectedImage != null)
                        ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: kIsWeb
                              ? Image.network(
                                  selectedImage!.path,
                                  height: 260,
                                  width: double.infinity,
                                  fit: BoxFit.contain,
                                  errorBuilder: (_, _, _) =>
                                      const Text('Image preview unavailable'),
                                )
                              : Image.file(
                                  File(selectedImage!.path),
                                  height: 260,
                                  width: double.infinity,
                                  fit: BoxFit.contain,
                                ),
                        )
                      else
                        Container(
                          height: 220,
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: const Color(0xFFF0F5F5),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.image_outlined,
                                  size: 54, color: Colors.teal),
                              SizedBox(height: 8),
                              Text('No image selected'),
                            ],
                          ),
                        ),
                      const SizedBox(height: 16),
                      Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: [
                          OutlinedButton.icon(
                            onPressed: chooseImage,
                            icon: const Icon(Icons.upload_file),
                            label: const Text('Choose image'),
                          ),
                          FilledButton.icon(
                            onPressed:
                                predicting || !apiOnline ? null : predictImage,
                            icon: predicting
                                ? const SizedBox(
                                    width: 18,
                                    height: 18,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2),
                                  )
                                : const Icon(Icons.analytics_outlined),
                            label: Text(
                                predicting ? 'Analyzing...' : 'Predict level'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // Result card
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Latest prediction',
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          const Icon(Icons.water_drop,
                              color: Colors.teal, size: 36),
                          const SizedBox(width: 14),
                          Expanded(
                            child: Text(
                              prettyLabel(predictionLabel),
                              style: const TextStyle(
                                  fontSize: 22, fontWeight: FontWeight.bold),
                            ),
                          ),
                          Text(
                            percent,
                            style: const TextStyle(
                                fontSize: 20, fontWeight: FontWeight.bold),
                          ),
                        ],
                      ),
                      if (predictionLabel != null) ...[
                        const SizedBox(height: 12),
                        Text(
                          'Model confidence: $percent. Verify the result visually; this is a prototype and not a clinical alarm.',
                          style: const TextStyle(color: Colors.black54),
                        ),
                      ],
                      if (predictionTime != null) ...[
                        const SizedBox(height: 8),
                        Text(
                          'Time: $predictionTime',
                          style: const TextStyle(
                              color: Colors.black45, fontSize: 12),
                        ),
                      ],
                    ],
                  ),
                ),
              ),

              if (errorMessage != null) ...[
                const SizedBox(height: 12),
                Text(
                  errorMessage!,
                  style: const TextStyle(color: Colors.red),
                ),
              ],

              const SizedBox(height: 20),
              const Text(
                'Prototype only · Not for clinical use',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.black45, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final bool online;
  final bool loading;

  const _StatusChip({required this.online, required this.loading});

  @override
  Widget build(BuildContext context) {
    final color = online ? Colors.teal : Colors.redAccent;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (loading)
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          else
            Icon(Icons.circle, size: 9, color: color),
          const SizedBox(width: 8),
          Text(
            loading ? 'Checking' : online ? 'Online' : 'Offline',
            style: TextStyle(color: color, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}