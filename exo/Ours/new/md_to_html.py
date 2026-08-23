#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Markdown to styled HTML for easy PDF export
"""
import markdown

# Read the markdown file
with open('TECHNICAL_ROADMAP.md', 'r', encoding='latin-1') as f:
    md_content = f.read()
# Convert to UTF-8
md_content = md_content.encode('latin-1').decode('utf-8', errors='ignore')

# Convert markdown to HTML
html_content = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code']
)

# Create a full HTML document with styling
full_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hand Pose Estimation - Technical Roadmap</title>
    <style>
        @page {
            size: A4;
            margin: 2cm;
        }
        
        body {
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-size: 12pt;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
            page-break-after: avoid;
        }
        
        h2 {
            color: #2980b9;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 25px;
            page-break-after: avoid;
        }
        
        h3 {
            color: #3498db;
            margin-top: 20px;
            page-break-after: avoid;
        }
        
        h4 {
            color: #16a085;
            margin-top: 15px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
        }
        
        th, td {
            border: 1px solid #bdc3c7;
            padding: 10px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 10pt;
            page-break-inside: avoid;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 10pt;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-left: 0;
            color: #7f8c8d;
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-radius: 0 5px 5px 0;
        }
        
        ul, ol {
            padding-left: 25px;
        }
        
        li {
            margin: 8px 0;
        }
        
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }
        
        @media print {
            body {
                font-size: 11pt;
            }
            
            pre {
                font-size: 9pt;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        }
    </style>
</head>
<body>
""" + html_content + """
</body>
</html>
"""

# Write the HTML file
with open('TECHNICAL_ROADMAP.html', 'w', encoding='utf-8') as f:
    f.write(full_html)

print("HTML file generated: TECHNICAL_ROADMAP.html")
print("Open it in browser and press Ctrl+P to save as PDF")
