import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MaizeYieldApp());
}

class MaizeYieldApp extends StatelessWidget {
  const MaizeYieldApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Maize Yield Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.green,
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  // ---------------------------------------------------------------------
  // IMPORTANT: point this at your deployed API before recording your demo.
  // For local testing on the iOS Simulator, 127.0.0.1 reaches your Mac's
  // localhost directly (no special IP needed, unlike the Android emulator).
  // ---------------------------------------------------------------------
static const String apiBaseUrl = "https://maize-yield-api-3626.onrender.com";
  final _formKey = GlobalKey<FormState>();

  final _regionController = TextEditingController();
  final _soilTypeController = TextEditingController();
  final _rainfallController = TextEditingController();
  final _temperatureController = TextEditingController();
  final _fertilizerController = TextEditingController();
  final _irrigationController = TextEditingController();
  final _weatherController = TextEditingController();
  final _daysToHarvestController = TextEditingController();

  bool _isLoading = false;
  String? _resultText;
  bool _resultIsError = false;

  @override
  void dispose() {
    _regionController.dispose();
    _soilTypeController.dispose();
    _rainfallController.dispose();
    _temperatureController.dispose();
    _fertilizerController.dispose();
    _irrigationController.dispose();
    _weatherController.dispose();
    _daysToHarvestController.dispose();
    super.dispose();
  }

  bool? _parseBool(String raw) {
    final v = raw.trim().toLowerCase();
    if (v == 'true' || v == 'yes' || v == '1') return true;
    if (v == 'false' || v == 'no' || v == '0') return false;
    return null;
  }

  Future<void> _submitPrediction() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _resultText = null;
      _resultIsError = false;
    });

    final fertilizerBool = _parseBool(_fertilizerController.text);
    final irrigationBool = _parseBool(_irrigationController.text);

    final body = jsonEncode({
      "Region": _regionController.text.trim(),
      "Soil_Type": _soilTypeController.text.trim(),
      "Rainfall_mm": double.tryParse(_rainfallController.text.trim()),
      "Temperature_Celsius": double.tryParse(_temperatureController.text.trim()),
      "Fertilizer_Used": fertilizerBool,
      "Irrigation_Used": irrigationBool,
      "Weather_Condition": _weatherController.text.trim(),
      "Days_to_Harvest": int.tryParse(_daysToHarvestController.text.trim()),
    });

    try {
      final response = await http
          .post(
            Uri.parse("$apiBaseUrl/predict"),
            headers: {"Content-Type": "application/json"},
            body: body,
          )
          .timeout(const Duration(seconds: 20));

      final decoded = jsonDecode(response.body);

      if (response.statusCode == 200) {
        final yieldValue = decoded["predicted_yield_tons_per_hectare"];
        setState(() {
          _resultText = "Predicted maize yield: $yieldValue tons/hectare";
          _resultIsError = false;
        });
      } else {
        // FastAPI validation errors come back as a list under "detail"
        String message;
        final detail = decoded["detail"];
        if (detail is List) {
          message = detail
              .map((e) => "${(e["loc"] as List).last}: ${e["msg"]}")
              .join("\n");
        } else {
          message = detail?.toString() ?? "Request failed (${response.statusCode}).";
        }
        setState(() {
          _resultText = "Error:\n$message";
          _resultIsError = true;
        });
      }
    } catch (e) {
      setState(() {
        _resultText = "Error: could not reach the API.\n$e";
        _resultIsError = true;
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: TextFormField(
        controller: controller,
        keyboardType: keyboardType,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: const OutlineInputBorder(),
        ),
        validator: (value) {
          if (value == null || value.trim().isEmpty) {
            return "$label is required";
          }
          return null;
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Maize Yield Predictor"),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  "Enter field conditions to predict maize yield "
                  "(tons/hectare).",
                  style: TextStyle(fontSize: 14, color: Colors.black54),
                ),
                const SizedBox(height: 12),
                _buildTextField(
                  controller: _regionController,
                  label: "Region",
                  hint: "North / East / South / West",
                ),
                _buildTextField(
                  controller: _soilTypeController,
                  label: "Soil Type",
                  hint: "Clay / Sandy / Loam / Silt / Peaty / Chalky",
                ),
                _buildTextField(
                  controller: _rainfallController,
                  label: "Rainfall (mm)",
                  hint: "e.g. 650",
                  keyboardType: const TextInputType.numberWithOptions(decimal: true),
                ),
                _buildTextField(
                  controller: _temperatureController,
                  label: "Temperature (Celsius)",
                  hint: "e.g. 23.5",
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                    signed: true,
                  ),
                ),
                _buildTextField(
                  controller: _fertilizerController,
                  label: "Fertilizer Used",
                  hint: "true / false",
                ),
                _buildTextField(
                  controller: _irrigationController,
                  label: "Irrigation Used",
                  hint: "true / false",
                ),
                _buildTextField(
                  controller: _weatherController,
                  label: "Weather Condition",
                  hint: "Sunny / Rainy / Cloudy",
                ),
                _buildTextField(
                  controller: _daysToHarvestController,
                  label: "Days to Harvest",
                  hint: "e.g. 104",
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _isLoading ? null : _submitPrediction,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text("Predict", style: TextStyle(fontSize: 16)),
                ),
                const SizedBox(height: 20),
                if (_resultText != null)
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: _resultIsError
                          ? Colors.red.shade50
                          : Colors.green.shade50,
                      border: Border.all(
                        color: _resultIsError ? Colors.red : Colors.green,
                      ),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      _resultText!,
                      style: TextStyle(
                        color: _resultIsError
                            ? Colors.red.shade900
                            : Colors.green.shade900,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
