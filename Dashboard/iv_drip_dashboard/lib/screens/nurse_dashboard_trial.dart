
import 'dart:async';
import 'dart:html' as html;
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
  List<dynamic> patients = [];
  final Map<String, Map<String, dynamic>> monitoringData = {};

  bool loading = true;
  bool refreshing = false;
  String? errorMessage;
  Timer? refreshTimer;

  // Opens the laptop webcam page for the selected patient.
  // The camera page is served separately on localhost:5500.
  void showLiveCamera(String patientName, String patientId) {
    final url =
        'http://localhost:5500/camera_patient_trial.html'
        '?patientId=${Uri.encodeComponent(patientId)}';

    html.window.open(url, '_blank');
  }

  @override
  void initState() {
    super.initState();
    loadPatients();

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

  Future<void> loadPatients() async {
    setState(() {
      loading = true;
      errorMessage = null;
    });

    try {
      final result = await AuthService.getPatients(widget.token);

      if (!mounted) return;

      setState(() {
        patients = result;
        loading = false;
      });

      await refreshMonitoring();
    } catch (e) {
      if (!mounted) return;

      setState(() {
        errorMessage = e.toString().replaceFirst('Exception: ', '');
        loading = false;
      });
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

      if (!mounted) return;

      setState(() {
        for (final result in results) {
          if (result != null) {
            monitoringData[result.key] = result.value;
          }
        }
      });
    } finally {
      refreshing = false;
    }
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

  Widget buildPatientCard(dynamic patient) {
    final patientId = patient['patient_id'].toString();
    final monitoring = monitoringData[patientId];

    final latestReading = monitoring?['latest_reading'];
    final deviceStatus =
        monitoring?['device_status']?.toString() ?? 'offline';

    final level = latestReading?['level_label']?.toString();
    final confidence = latestReading?['confidence'];

    final isEmpty = level == 'Empty';

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
            Text('Bed Number: ${patient['bed_number']}'),
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
                      : Colors.blue.withValues(alpha: 0.07),
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
                        color: isEmpty ? Colors.red : Colors.blue,
                      ),
                    ),
                    if (confidence != null) ...[
                      const SizedBox(height: 6),
                      Text(
                        'Confidence: ${(double.parse(confidence.toString()) * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                          color: Colors.black54,
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
                        'Empty level detected in the latest reading.',
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
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () => showLiveCamera(
                  patient['name'].toString(),
                  patientId,
                ),
                icon: const Icon(Icons.videocam),
                label: const Text('View Live Camera'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Nurse Dashboard'),
        actions: [
          IconButton(
            onPressed: loadPatients,
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh patients',
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
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1000),
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
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
              const SizedBox(height: 28),
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
                    padding: EdgeInsets.all(24),
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
            ],
          ),
        ),
      ),
    );
  }
}
