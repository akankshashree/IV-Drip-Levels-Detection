import 'dart:async';
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;
// ignore: avoid_web_libraries_in_flutter
import 'dart:ui_web' as ui_web;
import 'package:flutter/material.dart';
import '../auth_service.dart';
import 'login_page.dart';

class NurseDashboard extends StatefulWidget {
  final String token;
  final Map<String, dynamic> user;

  const NurseDashboard({
    super.key,
    required this.token,
    required this.user,
  });

  @override
  State<NurseDashboard> createState() => _NurseDashboardState();
}

class _NurseDashboardState extends State<NurseDashboard> {
  int _selectedTabIndex = 0; // 0: Monitoring, 1: Notifications
  String _selectedAlertFilter = 'all'; // 'all', 'unread', 'acknowledged', 'resolved'

  List<dynamic> patients = [];
  final Map<String, Map<String, dynamic>> monitoringData = {};
  List<dynamic> alerts = [];

  bool loading = true;
  bool refreshing = false;
  String? errorMessage;
  Timer? refreshTimer;

  // Default Raspberry Pi streaming address (user-configurable in UI)
  String piStreamHost = 'root.local:8000';
  final Map<String, String> selectedCameraSource = {}; // patientId -> 'pi' or 'webcam'

  // Track registered platform view IDs to avoid duplicate registration errors on rebuild
  static final Set<String> _registeredStreamViewIds = <String>{};

  static void _registerStreamViewFactory(String viewId, String streamUrl) {
    if (!_registeredStreamViewIds.contains(viewId)) {
      _registeredStreamViewIds.add(viewId);
      // ignore: undefined_prefixed_name
      ui_web.platformViewRegistry.registerViewFactory(
        viewId,
        (int id) {
          final imageElement = html.ImageElement()
            ..src = streamUrl
            ..style.width = '100%'
            ..style.height = '100%'
            ..style.objectFit = 'contain'
            ..style.border = 'none'
            ..style.backgroundColor = 'black';
          return imageElement;
        },
      );
    }
  }

  @override
  void initState() {
    super.initState();
    loadDashboardData();

    refreshTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) => refreshMonitoring(),
    );
  }

  @override
  void dispose() {
    refreshTimer?.cancel();
    super.dispose();
  }

  int get unreadAlertsCount => alerts.where((a) => a['status'] == 'new').length;

  String formatTimeOnly(dynamic isoString) {
    if (isoString == null) return '';
    final dt = DateTime.tryParse(isoString.toString());
    if (dt == null) return isoString.toString();
    final local = dt.toLocal().toString();
    return local.length >= 19 ? local.substring(11, 19) : local;
  }

  List<dynamic> get filteredAlerts {
    if (_selectedAlertFilter == 'unread') {
      return alerts.where((a) => a['status'] == 'new').toList();
    } else if (_selectedAlertFilter == 'acknowledged') {
      return alerts.where((a) => a['status'] == 'acknowledged').toList();
    } else if (_selectedAlertFilter == 'resolved') {
      return alerts.where((a) => a['status'] == 'resolved').toList();
    }
    return alerts;
  }

  Future<void> loadDashboardData() async {
    setState(() {
      loading = true;
      errorMessage = null;
    });

    try {
      final results = await Future.wait([
        AuthService.getPatients(widget.token),
        AuthService.getAlerts(widget.token),
      ]);

      if (!mounted) return;

      setState(() {
        patients = results[0];
        alerts = results[1];
        loading = false;
      });

      await refreshMonitoring();
    } catch (_) {
      if (!mounted) return;

      setState(() {
        errorMessage = 'Unable to load dashboard data. Please check connection and refresh.';
        loading = false;
      });
    }
  }

  Future<void> loadPatients() async {
    await loadDashboardData();
  }

  Future<void> loadAlerts() async {
    try {
      final fetchedAlerts = await AuthService.getAlerts(widget.token);
      if (!mounted) return;
      setState(() {
        alerts = fetchedAlerts;
      });
    } catch (_) {}
  }

  Future<void> acknowledgeAlert(int alertId) async {
    try {
      await AuthService.acknowledgeAlert(widget.token, alertId);
      await loadAlerts();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Alert acknowledged.'),
          backgroundColor: Colors.teal,
        ),
      );
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Could not acknowledge alert. Please try again.'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> resolveAlert(int alertId) async {
    try {
      await AuthService.resolveAlert(widget.token, alertId);
      await loadAlerts();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Alert resolved. IV container replaced.'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Could not resolve alert. Please try again.'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> triggerTestAlert(String patientId) async {
    try {
      await AuthService.sendTestAlert(widget.token, patientId);
      await loadAlerts();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Simulated empty-fluid condition for Patient $patientId.'),
          backgroundColor: Colors.teal,
        ),
      );
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Could not simulate alert. Please try again.'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> refreshMonitoring() async {
    if (refreshing || patients.isEmpty) return;

    refreshing = true;

    try {
      final results = await Future.wait(
        patients.map((patient) async {
          final patientId = patient['patient_id'].toString();

          try {
            final result = await AuthService.getPatientMonitoring(
              widget.token,
              patientId,
            );

            return MapEntry(patientId, result);
          } catch (_) {
            return null;
          }
        }),
      );

      final fetchedAlerts = await AuthService.getAlerts(widget.token);

      if (!mounted) return;

      setState(() {
        for (final result in results) {
          if (result != null) {
            monitoringData[result.key] = result.value;
          }
        }
        alerts = fetchedAlerts;
      });
    } catch (_) {
    } finally {
      refreshing = false;
    }
  }

  void showLiveCamera(String patientName, String patientId) {
    final hostController = TextEditingController(text: piStreamHost);
    String activeSource = selectedCameraSource[patientId] ?? 'pi';
    bool showSettings = false;

    showDialog(
      context: context,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            final monitoring = monitoringData[patientId];
            final latestReading = monitoring?['latest_reading'];
            final deviceStatus = monitoring?['device_status']?.toString() ?? 'offline';
            final level = latestReading?['level_label']?.toString();
            final confidence = latestReading?['confidence'];
            final isEmpty = level == 'Empty';

            // Sanitize host and construct standard video stream URL
            String cleanHost = piStreamHost.trim();
            if (cleanHost.startsWith('http://')) cleanHost = cleanHost.substring(7);
            if (cleanHost.startsWith('https://')) cleanHost = cleanHost.substring(8);
            if (cleanHost.endsWith('/video_feed')) cleanHost = cleanHost.substring(0, cleanHost.length - 11);
            if (cleanHost.endsWith('/')) cleanHost = cleanHost.substring(0, cleanHost.length - 1);
            final currentStreamUrl = 'http://$cleanHost/video_feed';

            final String streamViewId = 'pi-camera-stream-${cleanHost.replaceAll(RegExp(r'[^a-zA-Z0-9]'), '-')}';
            _registerStreamViewFactory(streamViewId, currentStreamUrl);

            return Dialog(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 820),
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(22),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Header
                      Row(
                        children: [
                          Icon(
                            activeSource == 'pi' ? Icons.videocam : Icons.laptop_chromebook,
                            color: const Color(0xFF0F766E),
                            size: 26,
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Live Monitoring — $patientName',
                                  style: const TextStyle(
                                    fontSize: 19,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  'Patient ID: $patientId | Source: ${activeSource == 'pi' ? 'Bedside Pi Camera (Bedside Monitor)' : 'Laptop Webcam (Demonstration Mode)'}',
                                  style: const TextStyle(color: Colors.black54, fontSize: 13),
                                ),
                              ],
                            ),
                          ),
                          IconButton(
                            onPressed: () => Navigator.pop(dialogContext),
                            icon: const Icon(Icons.close),
                            tooltip: 'Close',
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),

                      // Camera Source Selector (Toggle Buttons)
                      Container(
                        padding: const EdgeInsets.all(4),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade100,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.grey.shade300),
                        ),
                        child: Row(
                          children: [
                            Expanded(
                              child: InkWell(
                                onTap: () {
                                  setDialogState(() {
                                    activeSource = 'pi';
                                    selectedCameraSource[patientId] = 'pi';
                                  });
                                  setState(() {});
                                },
                                borderRadius: BorderRadius.circular(8),
                                child: Container(
                                  padding: const EdgeInsets.symmetric(vertical: 10),
                                  decoration: BoxDecoration(
                                    color: activeSource == 'pi' ? const Color(0xFF0F766E) : Colors.transparent,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  alignment: Alignment.center,
                                  child: Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(
                                        Icons.camera,
                                        size: 18,
                                        color: activeSource == 'pi' ? Colors.white : Colors.black87,
                                      ),
                                      const SizedBox(width: 8),
                                      Text(
                                        'Bedside Pi Camera (Bedside Monitor)',
                                        style: TextStyle(
                                          fontWeight: FontWeight.w600,
                                          fontSize: 13,
                                          color: activeSource == 'pi' ? Colors.white : Colors.black87,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 6),
                            Expanded(
                              child: InkWell(
                                onTap: () {
                                  setDialogState(() {
                                    activeSource = 'webcam';
                                    selectedCameraSource[patientId] = 'webcam';
                                  });
                                  setState(() {});
                                },
                                borderRadius: BorderRadius.circular(8),
                                child: Container(
                                  padding: const EdgeInsets.symmetric(vertical: 10),
                                  decoration: BoxDecoration(
                                    color: activeSource == 'webcam' ? const Color(0xFF0F766E) : Colors.transparent,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  alignment: Alignment.center,
                                  child: Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(
                                        Icons.laptop_chromebook,
                                        size: 18,
                                        color: activeSource == 'webcam' ? Colors.white : Colors.black87,
                                      ),
                                      const SizedBox(width: 8),
                                      Text(
                                        'Laptop Webcam (Demonstration Mode)',
                                        style: TextStyle(
                                          fontWeight: FontWeight.w600,
                                          fontSize: 13,
                                          color: activeSource == 'webcam' ? Colors.white : Colors.black87,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Camera Feed View Area
                      if (activeSource == 'pi') ...[
                        AspectRatio(
                          aspectRatio: 16 / 9,
                          child: Container(
                            decoration: BoxDecoration(
                              color: Colors.black,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            clipBehavior: Clip.antiAlias,
                            alignment: Alignment.center,
                            child: HtmlElementView(
                              key: ValueKey(streamViewId),
                              viewType: streamViewId,
                            ),
                          ),
                        ),
                        const SizedBox(height: 10),
                        // Collapsible Pi Host config
                        Row(
                          children: [
                            Text(
                              'Stream URL: $currentStreamUrl',
                              style: const TextStyle(fontSize: 12, color: Colors.black54),
                            ),
                            const Spacer(),
                            TextButton.icon(
                              onPressed: () {
                                setDialogState(() {
                                  showSettings = !showSettings;
                                });
                              },
                              icon: const Icon(Icons.settings, size: 16),
                              label: Text(showSettings ? 'Hide Address' : 'Change Pi Address', style: const TextStyle(fontSize: 12)),
                            ),
                          ],
                        ),
                        if (showSettings)
                          Padding(
                            padding: const EdgeInsets.only(top: 8),
                            child: Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    controller: hostController,
                                    decoration: const InputDecoration(
                                      labelText: 'Raspberry Pi Host:Port',
                                      hintText: 'root.local:8000 or 10.222.64.50:8000',
                                      border: OutlineInputBorder(),
                                      isDense: true,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                ElevatedButton(
                                  onPressed: () {
                                    String newHost = hostController.text.trim();
                                    if (newHost.startsWith('http://')) newHost = newHost.substring(7);
                                    if (newHost.startsWith('https://')) newHost = newHost.substring(8);
                                    if (newHost.endsWith('/video_feed')) newHost = newHost.substring(0, newHost.length - 11);
                                    if (newHost.endsWith('/')) newHost = newHost.substring(0, newHost.length - 1);
                                    setState(() {
                                      piStreamHost = newHost;
                                    });
                                    setDialogState(() {
                                      showSettings = false;
                                    });
                                  },
                                  child: const Text('Apply'),
                                ),
                              ],
                            ),
                          ),
                      ] else ...[
                        // Laptop Webcam Demonstration Console Launcher
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            color: Colors.teal.shade50,
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.teal.shade200),
                          ),
                          child: Column(
                            children: [
                              const Icon(
                                Icons.laptop_chromebook,
                                size: 54,
                                color: Color(0xFF0F766E),
                              ),
                              const SizedBox(height: 12),
                              const Text(
                                'Laptop Webcam Demonstration Mode Active',
                                style: TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.bold,
                                  color: Color(0xFF0F766E),
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'Webcam demonstration active for Patient $patientId ($patientName).\nAutomated MobileNetV2 predictions are updated at configurable intervals (5s or 10s).',
                                textAlign: TextAlign.center,
                                style: const TextStyle(fontSize: 13, color: Colors.black87),
                              ),
                              const SizedBox(height: 16),
                              Wrap(
                                spacing: 12,
                                runSpacing: 8,
                                alignment: WrapAlignment.center,
                                children: [
                                  ElevatedButton.icon(
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: const Color(0xFF0F766E),
                                      foregroundColor: Colors.white,
                                      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
                                    ),
                                    onPressed: () {
                                      final url = '${AuthService.baseUrl}/web/camera_patient_trial.html?patientId=${Uri.encodeComponent(patientId)}&token=${Uri.encodeComponent(widget.token)}';
                                      html.window.open(url, '_blank');
                                    },
                                    icon: const Icon(Icons.open_in_new),
                                    label: const Text('Open Demonstration Console'),
                                  ),
                                  OutlinedButton.icon(
                                    onPressed: () {
                                      setDialogState(() {
                                        activeSource = 'pi';
                                        selectedCameraSource[patientId] = 'pi';
                                      });
                                      setState(() {});
                                    },
                                    icon: const Icon(Icons.arrow_back),
                                    label: const Text('Switch back to Bedside Pi Camera'),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],

                      const Divider(height: 32),

                      // Corresponding IV Fluid-Level Predictions Section
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: isEmpty ? Colors.red.withValues(alpha: 0.08) : Colors.grey.shade100,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: isEmpty ? Colors.red.shade300 : Colors.grey.shade300,
                          ),
                        ),
                        child: Row(
                          children: [
                            CircleAvatar(
                              radius: 24,
                              backgroundColor: isEmpty
                                  ? Colors.red.withValues(alpha: 0.15)
                                  : const Color(0xFF0F766E).withValues(alpha: 0.15),
                              child: Icon(
                                isEmpty ? Icons.warning_amber : Icons.water_drop,
                                color: isEmpty ? Colors.red : const Color(0xFF0F766E),
                                size: 26,
                              ),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Current Fluid Level: ${level ?? 'Awaiting readings...'}',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: isEmpty ? Colors.red : Colors.black87,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    confidence != null
                                        ? 'Confidence: ${(double.parse(confidence.toString()) * 100).toStringAsFixed(1)}%'
                                          '${latestReading?['capture_interval'] != null ? ' | Interval: ${latestReading!['capture_interval']}s' : ''}'
                                          '${latestReading?['captured_at'] != null ? ' | Time: ${formatTimeOnly(latestReading!['captured_at'])}' : ''}'
                                          ' | Device: ${deviceStatus.toUpperCase()}'
                                        : 'Status: ${deviceStatus.toUpperCase()}',
                                    style: const TextStyle(fontSize: 12, color: Colors.black54),
                                  ),
                                ],
                              ),
                            ),
                            buildDeviceStatus(deviceStatus),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget buildDeviceStatus(String status) {
    final online = status == 'online';

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 7,
      ),
      decoration: BoxDecoration(
        color: online
            ? Colors.green.withValues(alpha: 0.12)
            : Colors.red.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.circle,
            size: 9,
            color: online ? Colors.green : Colors.red,
          ),
          const SizedBox(width: 7),
          Text(
            online ? 'Device Online' : 'Device Offline',
            style: TextStyle(
              color: online ? Colors.green : Colors.red,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget buildUrgentAlertBanner() {
    final newAlerts = alerts.where((a) => a['status'] == 'new').toList();
    if (newAlerts.isEmpty) return const SizedBox.shrink();

    return Column(
      children: newAlerts.map((alert) {
        final alertId = int.tryParse(alert['id'].toString()) ?? 0;
        final patientName = alert['patient_name'] ?? 'Patient';
        final bedNumber = alert['bed_number'] ?? '-';
        final patientId = alert['patient_id'] ?? '';
        final fluidLevel = alert['fluid_level']?.toString() ?? 'Empty';
        final alertType = alert['alert_type']?.toString() ?? 'IV_EMPTY';
        final alertTime = alert['created_at'] != null
            ? DateTime.tryParse(alert['created_at'])?.toLocal().toString().substring(11, 19) ?? alert['created_at']
            : 'Just now';
        final is50 = fluidLevel == '50%' || alertType == 'LOW_FLUID' || (alert['severity']?.toString().contains('WARN') ?? false);
        final bannerBg = is50 ? const Color(0xFFFFFBEB) : const Color(0xFFFEF2F2);
        final bannerBorder = is50 ? const Color(0xFFF59E0B) : const Color(0xFFEF4444);
        final bannerShadow = is50 ? Colors.amber.withValues(alpha: 0.1) : Colors.red.withValues(alpha: 0.1);
        final bannerIcon = is50 ? Icons.error_outline_rounded : Icons.warning_rounded;
        final iconColor = is50 ? const Color(0xFFD97706) : const Color(0xFFDC2626);
        final headerTitle = is50 ? 'LOW FLUID WARNING — IV FLUID LEVEL AT 50%' : 'URGENT CLINICAL ALERT — IV FLUID CONTAINER EMPTY';
        final badgeLabel = is50 ? 'LOW FLUID WARNING' : 'ACTION REQUIRED';
        final badgeBg = is50 ? const Color(0xFFD97706) : const Color(0xFFDC2626);
        final patientTextColor = is50 ? const Color(0xFF92400E) : const Color(0xFF7F1D1D);
        final bodyTextColor = is50 ? const Color(0xFFB45309) : const Color(0xFF991B1B);
        final alertMessage = alert['message']?.toString() ?? (is50
            ? 'IV fluid level has reached 50%. Please monitor the fluid level and prepare for replacement.'
            : 'The IV saline level has reached 0%. Please inspect the infusion line and replace the saline bottle immediately.');

        return Container(
          width: double.infinity,
          margin: const EdgeInsets.only(bottom: 16),
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: bannerBg,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: bannerBorder, width: 1.5),
            boxShadow: [
              BoxShadow(
                color: bannerShadow,
                blurRadius: 8,
                offset: const Offset(0, 3),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(bannerIcon, color: iconColor, size: 28),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      headerTitle,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: iconColor,
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: badgeBg,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      badgeLabel,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Text(
                'Patient: $patientName (ID: $patientId) | Bed Number: $bedNumber | Detected: $alertTime',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: patientTextColor,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                alertMessage,
                style: TextStyle(fontSize: 13, color: bodyTextColor),
              ),
              const SizedBox(height: 14),
              Wrap(
                spacing: 10,
                runSpacing: 8,
                children: [
                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: badgeBg,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    ),
                    onPressed: () => acknowledgeAlert(alertId),
                    icon: const Icon(Icons.check_circle_outline, size: 18),
                    label: const Text('Acknowledge Alert'),
                  ),
                  OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: patientTextColor,
                      side: BorderSide(color: iconColor),
                    ),
                    onPressed: () => showLiveCamera(patientName, patientId),
                    icon: const Icon(Icons.videocam, size: 18),
                    label: const Text('Check Camera Feed'),
                  ),
                  TextButton.icon(
                    onPressed: () {
                      setState(() {
                        _selectedTabIndex = 1;
                      });
                      loadAlerts();
                    },
                    icon: const Icon(Icons.notifications, size: 18),
                    label: const Text('View All Notifications'),
                  ),
                ],
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  Widget buildPatientCard(dynamic patient) {
    final patientId = patient['patient_id'].toString();
    final monitoring = monitoringData[patientId];

    final latestReading = monitoring?['latest_reading'];
    final deviceStatus =
        monitoring?['device_status']?.toString() ?? 'offline';

    final level = latestReading?['level_label']?.toString();
    final confidence = latestReading?['confidence'];

    final isEmpty = level == 'Empty';
    final activeSource = selectedCameraSource[patientId] ?? 'pi';

    return Card(
      margin: const EdgeInsets.only(bottom: 18),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const CircleAvatar(
                  radius: 24,
                  child: Icon(Icons.person),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        patient['name'].toString(),
                        style: const TextStyle(
                          fontSize: 19,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Patient ID: $patientId',
                        style: const TextStyle(color: Colors.black54),
                      ),
                    ],
                  ),
                ),
                buildDeviceStatus(deviceStatus),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Text('Bed Number: ${patient['bed_number']}'),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        activeSource == 'pi' ? Icons.camera : Icons.laptop_chromebook,
                        size: 13,
                        color: Colors.black87,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        activeSource == 'pi' ? 'Bedside Monitor: Pi Camera' : 'Demo Mode: Webcam',
                        style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const Divider(height: 28),
            const Row(
              children: [
                Icon(Icons.monitor_heart_outlined),
                SizedBox(width: 8),
                Text(
                  'IV Fluid Monitoring',
                  style: TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            if (latestReading == null)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Icon(
                        Icons.hourglass_empty,
                        size: 35,
                        color: Colors.grey,
                      ),
                      SizedBox(height: 8),
                      Text('No readings received yet'),
                    ],
                  ),
                ),
              )
            else ...[
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: isEmpty
                      ? Colors.red.withValues(alpha: 0.08)
                      : Colors.teal.withValues(alpha: 0.07),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    Text(
                      'Latest IV Fluid Level',
                      style: TextStyle(
                        color: Colors.grey.shade700,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      level ?? 'Unknown',
                      style: TextStyle(
                        fontSize: 34,
                        fontWeight: FontWeight.bold,
                        color: isEmpty ? Colors.red : const Color(0xFF0F766E),
                      ),
                    ),
                    if (confidence != null) ...[
                      const SizedBox(height: 6),
                      Text(
                        'Confidence: ${(double.parse(confidence.toString()) * 100).toStringAsFixed(1)}%'
                        '${latestReading?['capture_interval'] != null ? ' • Interval: ${latestReading!['capture_interval']}s' : ''}'
                        '${latestReading?['captured_at'] != null ? ' • Time: ${formatTimeOnly(latestReading!['captured_at'])}' : ''}',
                        style: const TextStyle(
                          color: Colors.black54,
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              const SizedBox(height: 12),
              if (isEmpty)
                const Row(
                  children: [
                    Icon(Icons.warning_amber, color: Colors.red),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Empty level detected in recent readings. Alert protocol active.',
                        style: TextStyle(
                          color: Colors.red,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
              if (deviceStatus == 'offline')
                const Padding(
                  padding: EdgeInsets.only(top: 8),
                  child: Row(
                    children: [
                      Icon(Icons.wifi_off, color: Colors.orange),
                      SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'No recent device reading. The displayed level may be outdated.',
                          style: TextStyle(color: Colors.orange),
                        ),
                      ),
                    ],
                  ),
                ),
            ],

            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => showLiveCamera(
                      patient['name'].toString(),
                      patientId,
                    ),
                    icon: Icon(
                      activeSource == 'pi' ? Icons.videocam : Icons.laptop_chromebook,
                    ),
                    label: Text(
                      'Live Camera (${activeSource == 'pi' ? 'Pi Camera' : 'Demo Webcam'})',
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton(
                  tooltip: 'Simulate Empty Alert for testing',
                  onPressed: () => triggerTestAlert(patientId),
                  icon: const Icon(Icons.add_alert_outlined, color: Colors.amber),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget buildNotificationCard(dynamic alert) {
    final alertId = int.tryParse(alert['id'].toString()) ?? 0;
    final status = alert['status']?.toString() ?? 'new';
    final severity = alert['severity']?.toString() ?? 'CRITICAL';
    final patientName = alert['patient_name'] ?? 'Patient';
    final bedNumber = alert['bed_number'] ?? '-';
    final patientId = alert['patient_id'] ?? '';
    final fluidLevel = alert['fluid_level']?.toString() ?? 'Empty';
    final emailStatus = alert['email_status']?.toString() ?? 'not configured';
    final alertType = alert['alert_type']?.toString() ?? 'IV_EMPTY';
    final alertMessage = alert['message']?.toString();
    final is50 = fluidLevel == '50%' || alertType == 'LOW_FLUID' || severity.contains('WARN') || severity.contains('LOW');
    final alertTime = alert['created_at'] != null
        ? DateTime.tryParse(alert['created_at'])?.toLocal().toString().substring(0, 19) ?? alert['created_at']
        : '-';

    Color statusBg;
    Color statusColor;
    String statusLabel;

    if (status == 'new') {
      statusBg = is50 ? const Color(0xFFFFFBEB) : const Color(0xFFFEF2F2);
      statusColor = is50 ? const Color(0xFFD97706) : const Color(0xFFDC2626);
      statusLabel = 'NEW / UNREAD';
    } else if (status == 'acknowledged') {
      statusBg = const Color(0xFFFFFBEB);
      statusColor = const Color(0xFFD97706);
      statusLabel = 'ACKNOWLEDGED';
    } else {
      statusBg = const Color(0xFFECFDF5);
      statusColor = const Color(0xFF059669);
      statusLabel = 'RESOLVED';
    }

    // Clean user-facing email dispatch indicator (no raw error traces or configuration details)
    IconData emailIcon;
    Color emailColor;
    String emailLabel;

    if (is50 || emailStatus == 'in_app_only') {
      emailIcon = Icons.notifications_active_outlined;
      emailColor = Colors.amber.shade800;
      emailLabel = 'In-App Notification (Low Fluid Warning)';
    } else if (emailStatus.startsWith('sent')) {
      emailIcon = Icons.mark_email_read_outlined;
      emailColor = Colors.teal.shade700;
      emailLabel = 'Alert Email: Dispatched to Nurse';
    } else if (emailStatus.startsWith('failed')) {
      emailIcon = Icons.schedule_outlined;
      emailColor = Colors.orange.shade800;
      emailLabel = 'Alert Email: Delivery Pending (Retrying)';
    } else {
      emailIcon = Icons.mail_outline;
      emailColor = Colors.grey.shade600;
      emailLabel = 'Alert Email: Standard Protocol';
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 14),
      elevation: status == 'new' ? 2 : 1,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(
          color: status == 'new'
              ? (is50 ? const Color(0xFFFDE68A) : const Color(0xFFFCA5A5))
              : Colors.grey.shade300,
          width: status == 'new' ? 1.5 : 1,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Top Row: Patient Info & Status Pill
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: statusBg,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(
                    status == 'resolved' ? Icons.check_circle : (is50 ? Icons.error_outline : Icons.warning_amber_rounded),
                    color: statusColor,
                    size: 26,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Flexible(
                            child: Text(
                              '$patientName (Bed $bedNumber)',
                              style: const TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.bold,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          const SizedBox(width: 8),
                          // Severity badge
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                            decoration: BoxDecoration(
                              color: is50 ? const Color(0xFFD97706) : const Color(0xFFDC2626),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              severity,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                          const SizedBox(width: 6),
                          // Status badge
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: statusBg,
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(color: statusColor.withValues(alpha: 0.5)),
                            ),
                            child: Text(
                              statusLabel,
                              style: TextStyle(
                                color: statusColor,
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Patient ID: $patientId | Detected: $alertTime',
                        style: const TextStyle(color: Colors.black54, fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),

            // Alert Message Body
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade200),
              ),
              child: Row(
                children: [
                  Icon(Icons.water_drop, color: is50 ? const Color(0xFFD97706) : const Color(0xFFDC2626), size: 18),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      alertMessage != null && alertMessage.isNotEmpty
                          ? alertMessage
                          : (is50
                              ? 'IV fluid level has reached 50%. Please monitor the fluid level and prepare for replacement.'
                              : 'IV Fluid Level: $fluidLevel (0% remaining). Immediate container replacement required.'),
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: is50 ? const Color(0xFF92400E) : const Color(0xFF7F1D1D),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 12),

            // Footer: User-friendly Email status and Action Buttons
            Row(
              children: [
                Icon(emailIcon, size: 16, color: emailColor),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    emailLabel,
                    style: TextStyle(fontSize: 12, color: emailColor, fontWeight: FontWeight.w500),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(width: 12),
                if (status == 'new')
                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF0F766E),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    ),
                    onPressed: () => acknowledgeAlert(alertId),
                    icon: const Icon(Icons.check, size: 16),
                    label: const Text('Acknowledge'),
                  ),
                if (status == 'acknowledged')
                  ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green.shade700,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    ),
                    onPressed: () => resolveAlert(alertId),
                    icon: const Icon(Icons.check_circle_outline, size: 16),
                    label: const Text('Resolve'),
                  ),
                if (status == 'resolved')
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.green.shade50,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.done_all, color: Colors.green.shade700, size: 16),
                        const SizedBox(width: 4),
                        Text(
                          'Resolved',
                          style: TextStyle(
                            color: Colors.green.shade700,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget buildNotificationsTab() {
    final unreadCount = alerts.where((a) => a['status'] == 'new').length;
    final ackCount = alerts.where((a) => a['status'] == 'acknowledged').length;
    final resCount = alerts.where((a) => a['status'] == 'resolved').length;
    final displayAlerts = filteredAlerts;

    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        // Header
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Notifications & Clinical Alerts',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Real-time log of IV fluid depletion detections and automated nurse alerts.',
                    style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
                  ),
                ],
              ),
            ),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF0F766E),
                foregroundColor: Colors.white,
              ),
              onPressed: loadAlerts,
              icon: const Icon(Icons.refresh, size: 18),
              label: const Text('Refresh'),
            ),
          ],
        ),
        const SizedBox(height: 18),

        // Filter Chips Row
        Wrap(
          spacing: 10,
          runSpacing: 8,
          children: [
            ChoiceChip(
              label: Text('All (${alerts.length})'),
              selected: _selectedAlertFilter == 'all',
              onSelected: (selected) {
                if (selected) setState(() => _selectedAlertFilter = 'all');
              },
            ),
            ChoiceChip(
              avatar: unreadCount > 0
                  ? const CircleAvatar(
                      backgroundColor: Colors.red,
                      radius: 8,
                    )
                  : null,
              label: Text('Unread / New ($unreadCount)'),
              selected: _selectedAlertFilter == 'unread',
              selectedColor: Colors.red.shade100,
              onSelected: (selected) {
                if (selected) setState(() => _selectedAlertFilter = 'unread');
              },
            ),
            ChoiceChip(
              label: Text('Acknowledged ($ackCount)'),
              selected: _selectedAlertFilter == 'acknowledged',
              selectedColor: Colors.amber.shade100,
              onSelected: (selected) {
                if (selected) setState(() => _selectedAlertFilter = 'acknowledged');
              },
            ),
            ChoiceChip(
              label: Text('Resolved ($resCount)'),
              selected: _selectedAlertFilter == 'resolved',
              selectedColor: Colors.green.shade100,
              onSelected: (selected) {
                if (selected) setState(() => _selectedAlertFilter = 'resolved');
              },
            ),
          ],
        ),
        const SizedBox(height: 18),

        // Notifications List
        if (displayAlerts.isEmpty)
          Card(
            elevation: 1,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 20),
              child: Column(
                children: [
                  Icon(
                    Icons.notifications_off_outlined,
                    size: 48,
                    color: Colors.grey.shade400,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _selectedAlertFilter == 'all'
                        ? 'No alerts recorded yet.'
                        : 'No alerts found in "${_selectedAlertFilter.toUpperCase()}" status.',
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Automated empty-fluid detections will appear here in real time.',
                    style: TextStyle(color: Colors.grey.shade600, fontSize: 13),
                  ),
                ],
              ),
            ),
          )
        else
          ...displayAlerts.map(buildNotificationCard),

        const SizedBox(height: 20),
        const Text(
          'Clinical alerts are retained in the patient audit record.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.black45, fontSize: 12),
        ),
      ],
    );
  }

  Widget buildMonitoringTab() {
    return ListView(
      padding: const EdgeInsets.all(24),
      children: [
        // Urgent Alert Banner (shown prominently if any unacknowledged alerts exist)
        buildUrgentAlertBanner(),

        Text(
          'Welcome, ${widget.user['name']}!',
          style: const TextStyle(
            fontSize: 26,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          widget.user['email'].toString(),
          style: const TextStyle(color: Colors.black54),
        ),
        const SizedBox(height: 24),

        const Text(
          'My Patients',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),

        if (loading)
          const Center(
            child: Padding(
              padding: EdgeInsets.all(30),
              child: CircularProgressIndicator(),
            ),
          ),

        if (errorMessage != null)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Text(
                errorMessage!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          ),

        if (!loading && errorMessage == null && patients.isEmpty)
          const Card(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                children: [
                  Icon(
                    Icons.people_outline,
                    size: 45,
                    color: Colors.grey,
                  ),
                  SizedBox(height: 12),
                  Text(
                    'No patients assigned to you yet.',
                    style: TextStyle(fontSize: 16),
                  ),
                ],
              ),
            ),
          ),

        if (!loading && errorMessage == null)
          ...patients.map(buildPatientCard),

        const Divider(height: 36),
        const Padding(
          padding: EdgeInsets.only(bottom: 24),
          child: Text(
            'Notice: This IV fluid monitoring system is an auxiliary clinical aid based on computer vision. '
            'Automated alerts are designed to support timely nursing attention and do not replace direct regular observation by trained clinical staff.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.black45, fontSize: 12, height: 1.5),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final unread = unreadAlertsCount;

    return Scaffold(
      appBar: AppBar(
        title: Text(_selectedTabIndex == 0 ? 'Nurse Dashboard — Monitoring' : 'Nurse Dashboard — Notifications'),
        actions: [
          IconButton(
            onPressed: loadDashboardData,
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh dashboard & alerts',
          ),
          TextButton.icon(
            onPressed: () {
              Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(
                  builder: (_) => const LoginPage(),
                ),
                (route) => false,
              );
            },
            icon: const Icon(Icons.logout),
            label: const Text('Logout'),
          ),
          const SizedBox(width: 12),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedTabIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedTabIndex = index;
          });
          if (index == 1) {
            loadAlerts();
          }
        },
        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.monitor_heart_outlined),
            selectedIcon: Icon(Icons.monitor_heart),
            label: 'Patient Monitoring',
          ),
          NavigationDestination(
            icon: unread > 0
                ? Badge(
                    label: Text('$unread'),
                    child: const Icon(Icons.notifications_outlined),
                  )
                : const Icon(Icons.notifications_outlined),
            selectedIcon: unread > 0
                ? Badge(
                    label: Text('$unread'),
                    child: const Icon(Icons.notifications),
                  )
                : const Icon(Icons.notifications),
            label: 'Notifications',
          ),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1000),
          child: _selectedTabIndex == 0 ? buildMonitoringTab() : buildNotificationsTab(),
        ),
      ),
    );
  }
}
