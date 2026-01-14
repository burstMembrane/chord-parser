import sys

from chord_parser import tab_parser

text = """[Verse]
Gm     C
Hello  world
"""
sheet = tab_parser.parse(text)

# Access sections
sys.stdout.write(sheet.sections[0].name + "\n")  # "Verse"

# Access blocks with chord-word attachments
block = sheet.sections[0].items[0]
for attachment in block.attachments.chord_to_word:
    sys.stdout.write(f"{attachment.chord.text} -> {attachment.word.text}\n")
