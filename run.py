{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1ab59dd9-57ed-4793-80ad-e37af56aaa1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import pandas as pd\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8a4ea2f4-b3dd-4b81-856c-f03d22980848",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_model(model_path):\n",
    "    # Load the trained model from the file\n",
    "    try:\n",
    "        loaded_model = joblib.load(model_path)\n",
    "        return loaded_model\n",
    "    except Exception as e:\n",
    "        print(f\"Error loading the model: {e}\")\n",
    "        sys.exit(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "366c5157-9693-4642-9628-911cd4f74362",
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_predictions(model, input_file, output_file):\n",
    "    try:\n",
    "        # Read the input data\n",
    "        test_data = pd.read_csv(input_file)\n",
    "\n",
    "        # Extract features from the test data\n",
    "        X_test = test_data.drop(columns=['uuid', 'datasetId', 'condition'])\n",
    "\n",
    "        # Make predictions using the loaded model\n",
    "        predictions = model.predict(X_test)\n",
    "\n",
    "        # Create a DataFrame with uuid and predicted HR values\n",
    "        results_df = pd.DataFrame({'uuid': test_data['uuid'], 'HR': predictions})\n",
    "\n",
    "        # Save the results to a CSV file\n",
    "        results_df.to_csv(output_file, index=False)\n",
    "        print(f\"Predictions saved to {output_file}\")\n",
    "    except Exception as e:\n",
    "        print(f\"Error making predictions: {e}\")\n",
    "        sys.exit(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "64bba587-f572-48b6-8805-6527a0253b53",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Error loading the model: [Errno 2] No such file or directory: '-f'\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "'tuple' object has no attribute 'tb_frame'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 4\u001b[0m, in \u001b[0;36mload_model\u001b[1;34m(model_path)\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m----> 4\u001b[0m     loaded_model \u001b[38;5;241m=\u001b[39m \u001b[43mjoblib\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodel_path\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m      5\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m loaded_model\n",
      "File \u001b[1;32m~\\anaconda3\\envs\\myenv\\Lib\\site-packages\\joblib\\numpy_pickle.py:650\u001b[0m, in \u001b[0;36mload\u001b[1;34m(filename, mmap_mode)\u001b[0m\n\u001b[0;32m    649\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 650\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mfilename\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mrb\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[0;32m    651\u001b[0m         \u001b[38;5;28;01mwith\u001b[39;00m _read_fileobject(f, filename, mmap_mode) \u001b[38;5;28;01mas\u001b[39;00m fobj:\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '-f'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mSystemExit\u001b[0m                                Traceback (most recent call last)",
      "    \u001b[1;31m[... skipping hidden 1 frame]\u001b[0m\n",
      "Cell \u001b[1;32mIn[7], line 11\u001b[0m\n\u001b[0;32m     10\u001b[0m \u001b[38;5;66;03m# Load the model\u001b[39;00m\n\u001b[1;32m---> 11\u001b[0m model \u001b[38;5;241m=\u001b[39m \u001b[43mload_model\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodel_path\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m     13\u001b[0m \u001b[38;5;66;03m# Make predictions and save results\u001b[39;00m\n",
      "Cell \u001b[1;32mIn[2], line 8\u001b[0m, in \u001b[0;36mload_model\u001b[1;34m(model_path)\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mError loading the model: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00me\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m----> 8\u001b[0m \u001b[43msys\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mexit\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m1\u001b[39;49m\u001b[43m)\u001b[49m\n",
      "\u001b[1;31mSystemExit\u001b[0m: 1",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "    \u001b[1;31m[... skipping hidden 1 frame]\u001b[0m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\interactiveshell.py:2097\u001b[0m, in \u001b[0;36mInteractiveShell.showtraceback\u001b[1;34m(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)\u001b[0m\n\u001b[0;32m   2094\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m exception_only:\n\u001b[0;32m   2095\u001b[0m     stb \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mAn exception has occurred, use \u001b[39m\u001b[38;5;124m%\u001b[39m\u001b[38;5;124mtb to see \u001b[39m\u001b[38;5;124m'\u001b[39m\n\u001b[0;32m   2096\u001b[0m            \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mthe full traceback.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m'\u001b[39m]\n\u001b[1;32m-> 2097\u001b[0m     stb\u001b[38;5;241m.\u001b[39mextend(\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mInteractiveTB\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_exception_only\u001b[49m\u001b[43m(\u001b[49m\u001b[43metype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   2098\u001b[0m \u001b[43m                                                     \u001b[49m\u001b[43mvalue\u001b[49m\u001b[43m)\u001b[49m)\n\u001b[0;32m   2099\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   2101\u001b[0m     \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mcontains_exceptiongroup\u001b[39m(val):\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:710\u001b[0m, in \u001b[0;36mListTB.get_exception_only\u001b[1;34m(self, etype, value)\u001b[0m\n\u001b[0;32m    702\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mget_exception_only\u001b[39m(\u001b[38;5;28mself\u001b[39m, etype, value):\n\u001b[0;32m    703\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Only print the exception type and message, without a traceback.\u001b[39;00m\n\u001b[0;32m    704\u001b[0m \n\u001b[0;32m    705\u001b[0m \u001b[38;5;124;03m    Parameters\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    708\u001b[0m \u001b[38;5;124;03m    value : exception value\u001b[39;00m\n\u001b[0;32m    709\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 710\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mListTB\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstructured_traceback\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43metype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mvalue\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:568\u001b[0m, in \u001b[0;36mListTB.structured_traceback\u001b[1;34m(self, etype, evalue, etb, tb_offset, context)\u001b[0m\n\u001b[0;32m    565\u001b[0m     chained_exc_ids\u001b[38;5;241m.\u001b[39madd(\u001b[38;5;28mid\u001b[39m(exception[\u001b[38;5;241m1\u001b[39m]))\n\u001b[0;32m    566\u001b[0m     chained_exceptions_tb_offset \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m\n\u001b[0;32m    567\u001b[0m     out_list \u001b[38;5;241m=\u001b[39m (\n\u001b[1;32m--> 568\u001b[0m         \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstructured_traceback\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    569\u001b[0m \u001b[43m            \u001b[49m\u001b[43metype\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    570\u001b[0m \u001b[43m            \u001b[49m\u001b[43mevalue\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    571\u001b[0m \u001b[43m            \u001b[49m\u001b[43m(\u001b[49m\u001b[43metb\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mchained_exc_ids\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m  \u001b[49m\u001b[38;5;66;43;03m# type: ignore\u001b[39;49;00m\n\u001b[0;32m    572\u001b[0m \u001b[43m            \u001b[49m\u001b[43mchained_exceptions_tb_offset\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    573\u001b[0m \u001b[43m            \u001b[49m\u001b[43mcontext\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    574\u001b[0m \u001b[43m        \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    575\u001b[0m         \u001b[38;5;241m+\u001b[39m chained_exception_message\n\u001b[0;32m    576\u001b[0m         \u001b[38;5;241m+\u001b[39m out_list)\n\u001b[0;32m    578\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m out_list\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:1435\u001b[0m, in \u001b[0;36mAutoFormattedTB.structured_traceback\u001b[1;34m(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[0;32m   1433\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   1434\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtb \u001b[38;5;241m=\u001b[39m etb\n\u001b[1;32m-> 1435\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mFormattedTB\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstructured_traceback\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1436\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43metype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mevalue\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43metb\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtb_offset\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnumber_of_lines_of_context\u001b[49m\n\u001b[0;32m   1437\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:1326\u001b[0m, in \u001b[0;36mFormattedTB.structured_traceback\u001b[1;34m(self, etype, value, tb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[0;32m   1323\u001b[0m mode \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmode\n\u001b[0;32m   1324\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m mode \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mverbose_modes:\n\u001b[0;32m   1325\u001b[0m     \u001b[38;5;66;03m# Verbose modes need a full traceback\u001b[39;00m\n\u001b[1;32m-> 1326\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mVerboseTB\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstructured_traceback\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m   1327\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43metype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mvalue\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtb\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtb_offset\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnumber_of_lines_of_context\u001b[49m\n\u001b[0;32m   1328\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1329\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m mode \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mMinimal\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[0;32m   1330\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m ListTB\u001b[38;5;241m.\u001b[39mget_exception_only(\u001b[38;5;28mself\u001b[39m, etype, value)\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:1173\u001b[0m, in \u001b[0;36mVerboseTB.structured_traceback\u001b[1;34m(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)\u001b[0m\n\u001b[0;32m   1164\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mstructured_traceback\u001b[39m(\n\u001b[0;32m   1165\u001b[0m     \u001b[38;5;28mself\u001b[39m,\n\u001b[0;32m   1166\u001b[0m     etype: \u001b[38;5;28mtype\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1170\u001b[0m     number_of_lines_of_context: \u001b[38;5;28mint\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m5\u001b[39m,\n\u001b[0;32m   1171\u001b[0m ):\n\u001b[0;32m   1172\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Return a nice text document describing the traceback.\"\"\"\u001b[39;00m\n\u001b[1;32m-> 1173\u001b[0m     formatted_exception \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mformat_exception_as_a_whole\u001b[49m\u001b[43m(\u001b[49m\u001b[43metype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mevalue\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43metb\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnumber_of_lines_of_context\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m   1174\u001b[0m \u001b[43m                                                           \u001b[49m\u001b[43mtb_offset\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m   1176\u001b[0m     colors \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mColors  \u001b[38;5;66;03m# just a shorthand + quicker name lookup\u001b[39;00m\n\u001b[0;32m   1177\u001b[0m     colorsnormal \u001b[38;5;241m=\u001b[39m colors\u001b[38;5;241m.\u001b[39mNormal  \u001b[38;5;66;03m# used a lot\u001b[39;00m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:1063\u001b[0m, in \u001b[0;36mVerboseTB.format_exception_as_a_whole\u001b[1;34m(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)\u001b[0m\n\u001b[0;32m   1060\u001b[0m \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(tb_offset, \u001b[38;5;28mint\u001b[39m)\n\u001b[0;32m   1061\u001b[0m head \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mprepare_header(\u001b[38;5;28mstr\u001b[39m(etype), \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mlong_header)\n\u001b[0;32m   1062\u001b[0m records \u001b[38;5;241m=\u001b[39m (\n\u001b[1;32m-> 1063\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_records\u001b[49m\u001b[43m(\u001b[49m\u001b[43metb\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnumber_of_lines_of_context\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtb_offset\u001b[49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mif\u001b[39;00m etb \u001b[38;5;28;01melse\u001b[39;00m []\n\u001b[0;32m   1064\u001b[0m )\n\u001b[0;32m   1066\u001b[0m frames \u001b[38;5;241m=\u001b[39m []\n\u001b[0;32m   1067\u001b[0m skipped \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m\n",
      "File \u001b[1;32m~\\AppData\\Roaming\\Python\\Python311\\site-packages\\IPython\\core\\ultratb.py:1131\u001b[0m, in \u001b[0;36mVerboseTB.get_records\u001b[1;34m(self, etb, number_of_lines_of_context, tb_offset)\u001b[0m\n\u001b[0;32m   1129\u001b[0m \u001b[38;5;28;01mwhile\u001b[39;00m cf \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m   1130\u001b[0m     \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m-> 1131\u001b[0m         mod \u001b[38;5;241m=\u001b[39m inspect\u001b[38;5;241m.\u001b[39mgetmodule(\u001b[43mcf\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtb_frame\u001b[49m)\n\u001b[0;32m   1132\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m mod \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m   1133\u001b[0m             mod_name \u001b[38;5;241m=\u001b[39m mod\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\n",
      "\u001b[1;31mAttributeError\u001b[0m: 'tuple' object has no attribute 'tb_frame'"
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    if len(sys.argv) != 3:\n",
    "        print(\"Usage: python run.py <model_path> <input_file> <output_file>\")\n",
    "        sys.exit(1)\n",
    "\n",
    "    model_path = sys.argv[1]\n",
    "    input_file = sys.argv[2]\n",
    "    output_file = \"stacking_model.joblib\"\n",
    "\n",
    "    # Load the model\n",
    "    model = load_model(model_path)\n",
    "\n",
    "    # Make predictions and save results\n",
    "    make_predictions(model, input_file, output_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb745b99-22d2-4183-aec2-f55806b6403c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
