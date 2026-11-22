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
  static const String _apiBaseUrl =
      "https://maize-yield-api-3626.onrender.com";
  static const Duration _requestTimeout = Duration(seconds: 20);

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
  bool _resultIsError = false;
  String? _resultText;

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
    final value = raw.trim().toLowerCase();
    if (value == 'true' || value == 'yes' || value == '1') return true;
    if (value == 'false' || value == 'no' || value == '0') return false;
    return null;
  }

  String _buildRequestBody() {
    return jsonEncode({
      "Region": _regionController.text.trim(),
      "Soil_Type": _soilTypeController.text.trim(),
      "Rainfall_mm": double.tryParse(_rainfallController.text.trim()),
      "Temperature_Celsius":
          double.tryParse(_temperatureController.text.trim()),
      "Fertilizer_Used": _parseBool(_fertilizerController.text),
      "Irrigation_Used": _parseBool(_irrigationController.text),
      "Weather_Condition": _weatherController.text.trim(),
      "Days_to_Harvest": int.tryParse(_daysToHarvestController.text.trim()),
    });
  }

  String _extractErrorMessage(dynamic decoded, int statusCode) {
    final detail = decoded["detail"];
    if (detail is List) {
      return detail
          .map((e) => "${(e["loc"] as List).last}: ${e["msg"]}")
          .join("\n");
    }
    return detail?.toString() ?? "Request failed ($statusCode).";
  }

  void _showResult(String text, {required bool isError}) {
    setState(() {
      _resultText = text;
      _resultIsError = isError;
    });
  }

  Future<void> _submitPrediction() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _resultText = null;
      _resultIsError = false;
    });

    try {
      final response = await http
          .post(
            Uri.parse("$_apiBaseUrl/predict"),
            headers: {"Content-Type": "application/json"},
            body: _buildRequestBody(),
          )
          .timeout(_requestTimeout);

      final decoded = jsonDecode(response.body);

      if (response.statusCode == 200) {
        final yieldValue = decoded["predicted_yield_tons_per_hectare"];
        _showResult(
          "Predicted maize yield: $yieldValue tons/hectare",
          isError: false,
        );
      } else {
        final message = _extractErrorMessage(decoded, response.statusCode);
        _showResult("Error:\n$message", isError: true);
      }
    } catch (e) {
      _showResult("Error: could not reach the API.\n$e", isError: true);
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

  Widget _buildFormFields() {
    return Column(
      children: [
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
      ],
    );
  }

  Widget _buildPredictButton() {
    return ElevatedButton(
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
    );
  }

  Widget _buildResultCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _resultIsError ? Colors.red.shade50 : Colors.green.shade50,
        border: Border.all(
          color: _resultIsError ? Colors.red : Colors.green,
        ),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        _resultText!,
        style: TextStyle(
          color: _resultIsError ? Colors.red.shade900 : Colors.green.shade900,
          fontWeight: FontWeight.w500,
        ),
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
                _buildFormFields(),
                const SizedBox(height: 20),
                _buildPredictButton(),
                const SizedBox(height: 20),
                if (_resultText != null) _buildResultCard(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}