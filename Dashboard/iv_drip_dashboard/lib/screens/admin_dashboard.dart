
import 'package:flutter/material.dart';
import '../auth_service.dart';
import 'login_page.dart';

class AdminDashboard extends StatefulWidget {
  final String token;
  final Map<String, dynamic> user;

  const AdminDashboard({
    super.key,
    required this.token,
    required this.user,
  });

  @override
  State<AdminDashboard> createState() => _AdminDashboardState();
}

class _AdminDashboardState extends State<AdminDashboard> {
  final nameController = TextEditingController();
  final emailController = TextEditingController();
  final passwordController = TextEditingController();

  final patientIdController = TextEditingController();
  final patientNameController = TextEditingController();
  final bedController = TextEditingController();

  bool creating = false;
  bool registeringPatient = false;
  bool success = false;
  String? message;
  String? patientMessage;

  List<dynamic> patients = [];
  List<dynamic> nurses = [];
  final Map<String, int?> selectedNurses = {};
  final Set<String> assigningPatients = {};

  @override
  void initState() {
    super.initState();
    loadData();
  }

  Future<void> loadData() async {
    await Future.wait([loadPatients(), loadNurses()]);
  }

  Future<void> loadNurses() async {
    try {
      final result = await AuthService.getNurses(widget.token);
      if (!mounted) return;

      setState(() {
        nurses = result;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        patientMessage = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  Future<void> createNurse() async {
    setState(() {
      creating = true;
      message = null;
    });

    try {
      await AuthService.createNurse(
        widget.token,
        nameController.text.trim(),
        emailController.text.trim(),
        passwordController.text,
      );

      await loadNurses();

      if (!mounted) return;
      setState(() {
        success = true;
        message = 'Nurse account created successfully!';
        nameController.clear();
        emailController.clear();
        passwordController.clear();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        success = false;
        message = e.toString().replaceFirst('Exception: ', '');
      });
    } finally {
      if (mounted) setState(() => creating = false);
    }
  }

  Future<void> registerPatient() async {
    setState(() {
      registeringPatient = true;
      patientMessage = null;
    });

    try {
      await AuthService.createPatient(
        widget.token,
        patientIdController.text.trim(),
        patientNameController.text.trim(),
        bedController.text.trim(),
      );

      patientIdController.clear();
      patientNameController.clear();
      bedController.clear();

      await loadPatients();

      if (!mounted) return;
      setState(() {
        patientMessage = 'Patient registered successfully!';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        patientMessage = e.toString().replaceFirst('Exception: ', '');
      });
    } finally {
      if (mounted) setState(() => registeringPatient = false);
    }
  }

  Future<void> loadPatients() async {
    try {
      final result = await AuthService.getPatients(widget.token);
      if (!mounted) return;

      setState(() {
        patients = result;

        for (final patient in result) {
          final patientId = patient['patient_id'].toString();
          final assignedId = patient['assigned_nurse_id'];

          selectedNurses[patientId] = assignedId == null
              ? null
              : (assignedId as num).toInt();
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        patientMessage = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  Future<void> assignNurse(String patientId) async {
    final nurseId = selectedNurses[patientId];

    if (nurseId == null) {
      setState(() {
        patientMessage = 'Please select a nurse first.';
      });
      return;
    }

    setState(() {
      assigningPatients.add(patientId);
      patientMessage = null;
    });

    try {
      await AuthService.assignPatient(
        widget.token,
        patientId,
        nurseId,
      );

      await loadPatients();

      if (!mounted) return;
      setState(() {
        patientMessage = 'Nurse assigned successfully!';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        patientMessage = e.toString().replaceFirst('Exception: ', '');
      });
    } finally {
      if (mounted) {
        setState(() {
          assigningPatients.remove(patientId);
        });
      }
    }
  }



  @override
  void dispose() {
    nameController.dispose();
    emailController.dispose();
    passwordController.dispose();
    patientIdController.dispose();
    patientNameController.dispose();
    bedController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Admin Dashboard'),
        actions: [
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
          constraints: const BoxConstraints(maxWidth: 850),
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
                widget.user['email'],
                style: const TextStyle(color: Colors.black54),
              ),
              const SizedBox(height: 28),

              const Text(
                'Create Nurse Account',
                style: TextStyle(
                  fontSize: 21,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),

              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      TextField(
                        controller: nameController,
                        decoration: const InputDecoration(
                          labelText: 'Nurse name',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 16),
                      TextField(
                        controller: emailController,
                        keyboardType: TextInputType.emailAddress,
                        decoration: const InputDecoration(
                          labelText: 'Nurse email',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 16),
                      TextField(
                        controller: passwordController,
                        obscureText: true,
                        decoration: const InputDecoration(
                          labelText: 'Temporary password',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: creating ? null : createNurse,
                          child: creating
                              ? const CircularProgressIndicator()
                              : const Text('Create Nurse Account'),
                        ),
                      ),
                      if (message != null) ...[
                        const SizedBox(height: 16),
                        Text(
                          message!,
                          style: TextStyle(
                            color: success ? Colors.green : Colors.red,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 32),

              const Text(
                'Register Patient',
                style: TextStyle(
                  fontSize: 21,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),

              Card(
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    children: [
                      TextField(
                        controller: patientIdController,
                        decoration: const InputDecoration(
                          labelText: 'Patient ID',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 16),
                      TextField(
                        controller: patientNameController,
                        decoration: const InputDecoration(
                          labelText: 'Patient name',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 16),
                      TextField(
                        controller: bedController,
                        decoration: const InputDecoration(
                          labelText: 'Bed number',
                          border: OutlineInputBorder(),
                        ),
                      ),
                      const SizedBox(height: 20),
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: registeringPatient
                              ? null
                              : registerPatient,
                          child: registeringPatient
                              ? const CircularProgressIndicator()
                              : const Text('Register Patient'),
                        ),
                      ),
                      if (patientMessage != null) ...[
                        const SizedBox(height: 16),
                        Text(patientMessage!),
                      ],
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 32),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Registered Patients',
                    style: TextStyle(
                      fontSize: 21,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  IconButton(
                    onPressed: loadData,
                    icon: const Icon(Icons.refresh),
                  ),
                ],
              ),

              const SizedBox(height: 12),

              if (patients.isEmpty)
                const Card(
                  child: Padding(
                    padding: EdgeInsets.all(20),
                    child: Text('No patients registered yet.'),
                  ),
                ),

              ...patients.map((patient) {
                final patientId = patient['patient_id'].toString();
                final assignedId = patient['assigned_nurse_id'];

                final assignedNurse = nurses.where(
                  (nurse) =>
                      nurse['id'] == assignedId,
                );

                final assignedName = assignedNurse.isEmpty
                    ? 'Not assigned'
                    : assignedNurse.first['name'].toString();

                return Card(
                  margin: const EdgeInsets.only(bottom: 14),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          patient['name'].toString(),
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text('Patient ID: $patientId'),
                        Text('Bed: ${patient['bed_number']}'),
                        Text('Assigned nurse: $assignedName'),
                        const SizedBox(height: 14),

                        DropdownButtonFormField<int>(
                          initialValue: selectedNurses[patientId],
                          decoration: const InputDecoration(
                            labelText: 'Select nurse',
                            border: OutlineInputBorder(),
                          ),
                          items: nurses.map((nurse) {
                            return DropdownMenuItem<int>(
                              value: (nurse['id'] as num).toInt(),
                              child: Text(nurse['name'].toString()),
                            );
                          }).toList(),
                          onChanged: (value) {
                            setState(() {
                              selectedNurses[patientId] = value;
                            });
                          },
                        ),

                        const SizedBox(height: 12),

                        SizedBox(
                          width: double.infinity,
                          child: FilledButton(
                            onPressed: assigningPatients.contains(patientId)
                                ? null
                                : () => assignNurse(patientId),
                            child: assigningPatients.contains(patientId)
                                ? const CircularProgressIndicator()
                                : const Text('Assign Nurse'),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }),
            ],
          ),
        ),
      ),
    );
  }
}
