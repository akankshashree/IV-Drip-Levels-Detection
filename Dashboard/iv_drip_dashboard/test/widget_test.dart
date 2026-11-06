import 'package:flutter_test/flutter_test.dart';
import 'package:iv_drip_dashboardd/main.dart';

void main() {
  testWidgets('IV Drip Monitor app loads', (WidgetTester tester) async {
    await tester.pumpWidget(const IVDripApp());

    expect(find.text('IV Drip Monitor'), findsOneWidget);
    expect(find.text('Monitoring Dashboard'), findsOneWidget);
  });
}