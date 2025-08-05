#!/usr/bin/env python3
"""
Test script for the database functionality
Populates the database with sample data to test the UI
"""

import sys
import os
sys.path.append('/workspace')

from database_manager import initialize_database
import time
import random

def populate_sample_data():
    """Populate the database with sample data"""
    try:
        print("🔧 Initializing database...")
        db = initialize_database()
        
        print("📊 Adding sample templates...")
        # Add sample templates
        templates = [
            {
                'headers': ['Name', 'Age', 'City', 'Salary'],
                'row_count': 10,
                'column_count': 4,
                'confidence_score': 0.95
            },
            {
                'headers': ['Product', 'Price', 'Quantity', 'Total'],
                'row_count': 8,
                'column_count': 4,
                'confidence_score': 0.88
            },
            {
                'headers': ['Date', 'Description', 'Amount'],
                'row_count': 15,
                'column_count': 3,
                'confidence_score': 0.92
            }
        ]
        
        for template in templates:
            success = db.save_learned_template(template)
            print(f"   Template saved: {success}")
        
        print("✏️ Adding header corrections...")
        # Add sample header corrections
        corrections = [
            ("nm", "Name"),
            ("ag", "Age"),
            ("cty", "City"),
            ("sal", "Salary"),
            ("prod", "Product"),
            ("prc", "Price"),
            ("qty", "Quantity"),
            ("tot", "Total"),
            ("dt", "Date"),
            ("desc", "Description"),
            ("amt", "Amount")
        ]
        
        for original, corrected in corrections:
            success = db.save_header_correction(original, corrected)
            print(f"   Correction saved: {original} -> {corrected}: {success}")
        
        print("📈 Adding extraction statistics...")
        # Add sample extraction stats
        file_names = ["invoice_001.pdf", "report_002.pdf", "contract_003.pdf", "data_004.pdf", "summary_005.pdf"]
        
        for i, file_name in enumerate(file_names):
            extraction_id = db.save_extraction_stats(
                file_name=file_name,
                file_hash=f"hash_{i}_{int(time.time())}",
                tables_extracted=random.randint(1, 5),
                extraction_method="adaptive",
                success_rate=random.uniform(0.7, 1.0),
                processing_time=random.uniform(0.5, 3.0)
            )
            print(f"   Extraction saved: {file_name} (ID: {extraction_id})")
        
        print("\n📊 Current database statistics:")
        stats = db.get_learning_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Sample data population completed!")
        print("🌐 You can now run the application and see the data:")
        print("   python3 run_adaptive_app.py")
        
    except Exception as e:
        print(f"❌ Error populating sample data: {e}")
        return False
    
    return True

def test_database_operations():
    """Test basic database operations"""
    try:
        print("🧪 Testing database operations...")
        db = initialize_database()
        
        # Test template saving
        test_template = {
            'headers': ['Test1', 'Test2'],
            'row_count': 2,
            'column_count': 2,
            'confidence_score': 0.5
        }
        
        result = db.save_learned_template(test_template)
        print(f"   Template save test: {'✅ PASS' if result else '❌ FAIL'}")
        
        # Test header correction
        result = db.save_header_correction("tst", "Test")
        print(f"   Header correction test: {'✅ PASS' if result else '❌ FAIL'}")
        
        # Test statistics
        stats = db.get_learning_statistics()
        has_stats = isinstance(stats, dict) and 'templates_learned' in stats
        print(f"   Statistics test: {'✅ PASS' if has_stats else '❌ FAIL'}")
        
        print("🧪 Database tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def main():
    """Main function"""
    print("🗄️ Database Test and Population Script")
    print("=" * 50)
    
    # Test database operations
    test_success = test_database_operations()
    
    if test_success:
        print("\n" + "=" * 50)
        # Populate with sample data
        populate_success = populate_sample_data()
        
        if populate_success:
            print("\n🎉 All operations completed successfully!")
        else:
            print("\n❌ Sample data population failed")
            return 1
    else:
        print("\n❌ Database tests failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())