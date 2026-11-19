"""Export the completed four-condition CALVIN matrix as an editable spreadsheet.

CSV and XLSX contain the same compact table: task rows, three columns per
condition/axis, and a final averages row. Step means match run_evaluation.py:
policy calls among successful trials, blank when no trial succeeded.
"""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

CONDITIONS = (
    ('overloaded_base', 'Base'),
    ('overloaded_distilled', 'Distilled'),
    ('overloaded_base_steered', 'Base + steering'),
    ('overloaded_distilled_steered', 'Distilled + steering'),
)
AXES = ('P0', 'P1', 'P2', 'P3', 'P4')
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'


def spreadsheet_round(value):
    return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def column_name(number):
    name = ''
    while number:
        number, remainder = divmod(number-1, 26)
        name = chr(65+remainder)+name
    return name


def build_table(root, *, backbone='3dda'):
    if backbone not in ('3dda', 'pi05'):
        raise ValueError(f'Unsupported backbone: {backbone}')
    prefix = 'pi05_' if backbone == 'pi05' else ''
    conditions = tuple((prefix + condition, label) for condition, label in CONDITIONS)
    results = {(condition, axis): json.loads((root/axis/f'{condition}.json').read_text())
               for condition, _ in conditions for axis in AXES}
    tasks = list(results[(prefix + 'overloaded_base', 'P0')]['tasks'])
    if len(tasks) != 34:
        raise ValueError('Expected 34 CALVIN tasks')
    for (condition, axis), result in results.items():
        if set(result['tasks']) != set(tasks):
            raise ValueError(f'Task mismatch in {condition}/{axis}')
        for task in tasks:
            trials = result['tasks'][task]['episodes']
            if [e['episode_idx'] for e in trials] != list(range(30)):
                raise ValueError(f'Expected exactly 30 ordered trials: {condition}/{axis}/{task}')
    header = ['Task']
    for _, label in conditions:
        for axis in AXES:
            prefix = f'{axis} {label}'
            header.extend([f'{prefix} successes (out of 30)',
                           f'{prefix} percentage (%)', f'{prefix} avg steps (successful)'])
    table = [header]
    cached = [header]
    for row_index, task in enumerate(tasks, 2):
        row, values = [task], [task]
        for condition, _ in conditions:
            for axis in AXES:
                trials = results[(condition, axis)]['tasks'][task]['episodes']
                successes = [e for e in trials if e['success']]
                count = len(successes)
                steps = spreadsheet_round(Decimal(sum(e['steps'] for e in successes))/count) if successes else ''
                count_column = column_name(len(row)+1)
                row.extend([count, f'=ROUND({count_column}{row_index}/30*100,2)', steps])
                values.extend([count, spreadsheet_round(Decimal(count)*100/30), steps])
        table.append(row)
        cached.append(values)
    last_task_row = len(table)
    averages, average_values = ['Average'], ['Average']
    for column in range(2, len(header)+1):
        letter = column_name(column)
        averages.append(f'=IFERROR(ROUND(AVERAGE({letter}2:{letter}{last_task_row}),2),"")')
        numbers = [row[column-1] for row in cached[1:] if isinstance(row[column-1], (int, float))]
        average_values.append(spreadsheet_round(sum(Decimal(str(v)) for v in numbers)/len(numbers)) if numbers else '')
    table.append(averages)
    cached.append(average_values)
    return table, cached


def write_xlsx(path, table, cached):
    """Small dependency-free OOXML workbook with editable inputs and formulas."""
    ET.register_namespace('', NS)
    def node(parent, tag, attributes=None, value=None):
        element = ET.SubElement(parent, f'{{{NS}}}{tag}', attributes or {})
        if value is not None:
            element.text = str(value)
        return element
    def xml(element):
        return ET.tostring(element, encoding='utf-8', xml_declaration=True)
    sheet = ET.Element(f'{{{NS}}}worksheet')
    node(sheet, 'dimension', {'ref': f'A1:{column_name(len(table[0]))}{len(table)}'})
    views = node(sheet, 'sheetViews')
    view = node(views, 'sheetView', {'workbookViewId': '0', 'zoomScale': '85'})
    node(view, 'pane', {'xSplit': '1', 'ySplit': '1', 'topLeftCell': 'B2',
                        'activePane': 'bottomRight', 'state': 'frozen'})
    node(view, 'selection', {'pane': 'bottomRight', 'activeCell': 'B2', 'sqref': 'B2'})
    columns = node(sheet, 'cols')
    node(columns, 'col', {'min': '1', 'max': '1', 'width': '31', 'customWidth': '1'})
    node(columns, 'col', {'min': '2', 'max': str(len(table[0])), 'width': '17', 'customWidth': '1'})
    data = node(sheet, 'sheetData')
    for row_index, row in enumerate(table, 1):
        attributes = {'r': str(row_index)}
        if row_index == 1:
            attributes.update(ht='80', customHeight='1')
        row_node = node(data, 'row', attributes)
        for column, value in enumerate(row, 1):
            if row_index == 1:
                style = 3 if column == 1 else 4+(column-2)//15
            elif row_index == len(table):
                style = 8 if column == 1 else (10 if (column-2)%3 == 1 else 9)
            else:
                style = 0 if column == 1 or (column-2)%3 == 0 else (2 if (column-2)%3 == 1 else 1)
            cell = node(row_node, 'c', {'r': f'{column_name(column)}{row_index}', 's': str(style)})
            if isinstance(value, str) and value.startswith('='):
                node(cell, 'f', value=value[1:])
                result = cached[row_index-1][column-1]
                if result != '':
                    node(cell, 'v', value=result)
            elif isinstance(value, (int, float)):
                node(cell, 'v', value=value)
            elif value:
                cell.set('t', 'inlineStr')
                node(node(cell, 'is'), 't', value=value)
    node(sheet, 'autoFilter', {'ref': f'A1:{column_name(len(table[0]))}{len(table)-1}'})
    validations = node(sheet, 'dataValidations', {'count': '1'})
    validation = node(validations, 'dataValidation', {
        'type': 'whole', 'operator': 'between', 'allowBlank': '1', 'showErrorMessage': '1',
        'errorTitle': 'Success count', 'error': 'Enter a whole number between 0 and 30.',
        'sqref': ' '.join(f'{column_name(c)}2:{column_name(c)}{len(table)-1}'
                         for c in range(2, len(table[0])+1, 3))})
    node(validation, 'formula1', value='0')
    node(validation, 'formula2', value='30')
    styles = ET.Element(f'{{{NS}}}styleSheet')
    formats = node(styles, 'numFmts', {'count': '1'})
    node(formats, 'numFmt', {'numFmtId': '164', 'formatCode': '0.00"%"'})
    fonts = node(styles, 'fonts', {'count': '2'})
    for bold in (False, True):
        font = node(fonts, 'font')
        node(font, 'sz', {'val': '11'})
        node(font, 'name', {'val': 'Calibri'})
        if bold:
            node(font, 'b')
    fills = node(styles, 'fills', {'count': '7'})
    for kind in ('none', 'gray125'):
        node(node(fills, 'fill'), 'patternFill', {'patternType': kind})
    for color in ('FFE2E8F0', 'FFDCEAF7', 'FFE1F0E4', 'FFFCE8D5', 'FFECE2F5'):
        fill = node(node(fills, 'fill'), 'patternFill', {'patternType': 'solid'})
        node(fill, 'fgColor', {'rgb': color})
        node(fill, 'bgColor', {'indexed': '64'})
    borders = node(styles, 'borders', {'count': '1'})
    border = node(borders, 'border')
    for side in ('left', 'right', 'top', 'bottom', 'diagonal'):
        node(border, side)
    base = node(styles, 'cellStyleXfs', {'count': '1'})
    node(base, 'xf', {'numFmtId': '0', 'fontId': '0', 'fillId': '0', 'borderId': '0'})
    formats = node(styles, 'cellXfs', {'count': '11'})
    definitions = [(0,0,0,False), (2,0,0,False), (164,0,0,False),
                   (0,1,2,True), (0,1,3,True), (0,1,4,True), (0,1,5,True), (0,1,6,True),
                   (0,1,2,False), (2,1,2,False), (164,1,2,False)]
    for number, font, fill, wrap in definitions:
        xf = node(formats, 'xf', {'numFmtId': str(number), 'fontId': str(font),
                                  'fillId': str(fill), 'borderId': '0', 'xfId': '0',
                                  'applyNumberFormat': '1', 'applyFill': '1', 'applyFont': '1',
                                  'applyAlignment': '1'})
        node(xf, 'alignment', {'vertical': 'center', 'wrapText': '1' if wrap else '0'})
    cell_styles = node(styles, 'cellStyles', {'count': '1'})
    node(cell_styles, 'cellStyle', {'name': 'Normal', 'xfId': '0', 'builtinId': '0'})
    parts = {
        '[Content_Types].xml': '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>''',
        '_rels/.rels': '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''',
        'xl/workbook.xml': f'''<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="CALVIN results" sheetId="1" r:id="rId1"/></sheets><calcPr calcId="0" calcMode="auto" fullCalcOnLoad="1" forceFullCalc="1"/></workbook>''',
        'xl/_rels/workbook.xml.rels': '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>''',
        'xl/worksheets/sheet1.xml': xml(sheet),
        'xl/styles.xml': xml(styles),
    }
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=('3dda', 'pi05'), default='3dda')
    parser.add_argument('--results', type=Path, required=True,
                        help='the --output-dir of run_overloaded_evaluation.py matrix')
    parser.add_argument('--output', type=Path, default=None,
                        help='CSV path (default: <results>/calvin_editable_results.csv); an .xlsx is written beside it')
    args = parser.parse_args()
    if args.output is None:
        args.output = args.results / 'calvin_editable_results.csv'
    table, cached = build_table(args.results, backbone=args.backbone)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    xlsx = args.output.with_suffix('.xlsx')
    if args.output.exists() or xlsx.exists():
        raise FileExistsError('Choose a fresh --output path to preserve manual spreadsheet corrections')
    with args.output.open('w', newline='', encoding='utf-8-sig') as stream:
        csv.writer(stream).writerows(table)
    write_xlsx(xlsx, table, cached)
    print(f'{args.output}\n{xlsx}\n34 tasks + averages; 61 columns; successful-trial policy-call means.')


if __name__ == '__main__':
    main()
